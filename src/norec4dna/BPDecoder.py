import os
from collections import deque
from typing import IO, Deque, Dict, List, Optional, Set, Tuple, Union

import numpy as np

from .Decoder import Decoder
from .distributions import Distribution
from .ErrorCorrection import ErrorCorrectionCallable, nocode
from .GEPP import GEPP
from .HeaderChunk import HeaderChunk
from .helper.RU10Helper import from_true_false_list
from .OnlineAuxPacket import OnlineAuxPacket
from .OnlinePacket import OnlinePacket
from .Packet import Packet
from .RU10IntermediatePacket import RU10IntermediatePacket
from .RU10Packet import RU10Packet


class BPDecoder(Decoder):
    """Belief-propagation (peeling) decoder shared by the LT, Online and RU10 BP decoders.

    Received packets are expressed over the source chunks (``addPacket``: auxiliary
    packets removed), registered (``_register_packet``) and peeled
    (``updatePackets`` / ``_peel``); ``solve`` adds inactivation decoding for what
    peeling cannot resolve.  For every received packet, the decoder keeps its
    original equation (``get_received_system``) and tracks which received packets
    were XORed into every solved chunk (``provenance_matrix``, the same information
    as ``GEPP.chunk_to_used_packets`` of a GEPP decoder).
    """

    def __init__(
        self,
        file: Optional[str] = None,
        error_correction: ErrorCorrectionCallable = nocode,
        use_headerchunk: bool = True,
        static_number_of_chunks: Optional[int] = None,
        use_method: bool = False,
    ):
        super().__init__(file=file)
        self.debug = False
        self.isPseudo: bool = False
        self.file: Optional[str] = file
        self.use_method: bool = use_method
        self.f: Optional[IO[bytes]] = None
        if file is not None:
            self.isFolder = os.path.isdir(file)
            if not self.isFolder:
                try:
                    self.f = open(file, "rb")
                except Exception:
                    self.f = None
        self.correct: int = 0
        self.corrupt: int = 0
        self.number_of_chunks: int = 1000000
        self.headerChunk: Optional[HeaderChunk] = None
        self.decodedPackets: Set[Packet] = set()
        self.queue: Deque[Packet] = deque()
        self.symbol_to_packets: Dict[int, Set[Packet]] = {}
        self.solved_symbols: Dict[int, Packet] = {}
        self.pseudoCount: int = 0
        self.repairBlockNumbers: Dict[int, Set[int]] = {}
        self.s: int = -1
        self.h: int = -1
        self.numberOfDecodedAuxBlocks: int = 0
        self.dist: Optional[Distribution] = None
        self.EOF: bool = False
        self.counter: Dict[int, int] = {}
        self.count: bool = False
        self.error_correction: ErrorCorrectionCallable = error_correction
        self.use_headerchunk: bool = use_headerchunk
        self.static_number_of_chunks: Optional[int] = static_number_of_chunks
        self.auxBlocks: Dict[int, Union[RU10IntermediatePacket, OnlineAuxPacket]] = {}
        self.degreeToPacket: Dict[int, Set[Packet]] = {}
        self.ldpcANDhalf: Dict[int, RU10IntermediatePacket] = {}
        # Received packets (reduced in place while peeling) and their original
        # equations in input order; packet p is bit p of the ``provenance`` of every
        # packet / solved symbol it was XORed into.
        self.packets: List[Packet] = []
        self.received_rows: List[Tuple[Set[int], np.ndarray]] = []

    def addPacket(self, packet: Union[Packet, RU10Packet, OnlinePacket]) -> None:
        """Express *packet* over the source chunks and register it as received."""
        removed = self.removeAndXorAuxPackets(packet)
        packet.set_used_packets(set(from_true_false_list(removed)))
        if self.static_number_of_chunks is None:
            self.number_of_chunks = packet.get_total_number_of_chunks()
        self._register_packet(packet, packet.get_used_packets())

    def _register_packet(self, packet: Packet, used: Set[int]) -> None:
        """Record a received packet (its original equation) and give it its provenance bit."""
        packet.provenance = 1 << len(self.received_rows)
        data = packet.get_data()
        self.received_rows.append(
            (
                set(used),
                np.frombuffer(bytes(data if data is not None else b""), dtype=np.uint8).copy(),
            )
        )
        self.packets.append(packet)

    def get_received_system(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (A, b) of all received packets in input order, as a GEPP decoder holds it."""
        a = np.zeros((len(self.received_rows), self.number_of_chunks), dtype=bool)
        for row, (used, _) in enumerate(self.received_rows):
            a[row, [u for u in used if u < self.number_of_chunks]] = True
        width = max((len(data) for _, data in self.received_rows), default=0)
        b = np.zeros((len(self.received_rows), width), dtype=np.uint8)
        for row, (_, data) in enumerate(self.received_rows):
            b[row, : len(data)] = data
        return a, b

    def provenance_matrix(self) -> np.ndarray:
        """Row i: the received packets (input order) whose XOR is the solved chunk i."""
        n = self.number_of_chunks
        width = max(n, len(self.received_rows))
        n_bytes = (width + 7) // 8
        matrix = np.zeros((n, width), dtype=bool)
        for i, obj in self.solved_symbols.items():
            prov = getattr(obj, "provenance", 0)
            if i < n and prov:
                bits = np.unpackbits(
                    np.frombuffer(prov.to_bytes(n_bytes, "little"), dtype=np.uint8),
                    bitorder="little",
                )
                matrix[i] = bits[:width].astype(bool)
        return matrix

    def updatePackets(self, packet: Packet) -> bool:
        deg = packet.get_degree()
        if deg == 0:
            return self.is_decoded()

        # Un-XOR already solved symbols from this incoming packet first
        for sym_id in list(packet.get_used_packets()):
            if sym_id in self.solved_symbols:
                packet.xor_and_remove_packet(self.solved_symbols[sym_id])

        deg = packet.get_degree()
        if deg == 0:
            return self.is_decoded()

        if deg == 1:
            self.queue.append(packet)
        else:
            for sym_id in packet.get_used_packets():
                if sym_id not in self.symbol_to_packets:
                    self.symbol_to_packets[sym_id] = set()
                self.symbol_to_packets[sym_id].add(packet)

        # Peeling only: the expensive inactivation step (if any) is deferred to
        # the explicit solve() call so the per-packet path stays near-linear.
        return self._peel()

    def _peel(self) -> bool:
        """Run one belief-propagation peeling pass (near-linear).

        Solves every symbol that can be resolved from degree-1 packets.  This is
        the cheap, incremental solver used on the per-packet hot path; the more
        expensive inactivation step (if any) is deferred to ``solve()``.
        """
        while len(self.queue) > 0:
            deg1_pkt = self.queue.popleft()
            if deg1_pkt.get_degree() != 1:
                # Reduce by known solved symbols
                for sym_id in list(deg1_pkt.get_used_packets()):
                    if sym_id in self.solved_symbols:
                        deg1_pkt.xor_and_remove_packet(self.solved_symbols[sym_id])
                if deg1_pkt.get_degree() != 1:
                    if deg1_pkt.get_degree() > 1:
                        for sym_id in deg1_pkt.get_used_packets():
                            if sym_id not in self.symbol_to_packets:
                                self.symbol_to_packets[sym_id] = set()
                            self.symbol_to_packets[sym_id].add(deg1_pkt)
                    continue

            [sym_id] = deg1_pkt.get_used_packets()
            if sym_id >= self.number_of_chunks:
                continue

            if sym_id in self.solved_symbols:
                continue

            # Store solved symbol
            self._store_solved_symbol(sym_id, deg1_pkt)

            # Propagate symbol_id to all unreduced packets containing symbol_id
            affected_packets = self.symbol_to_packets.pop(sym_id, set())
            for other_pkt in affected_packets:
                if other_pkt is deg1_pkt:
                    # Never XOR a packet with itself: this would zero the just-solved symbol.
                    continue
                if sym_id in other_pkt.get_used_packets():
                    other_pkt.xor_and_remove_packet(deg1_pkt)
                    new_deg = other_pkt.get_degree()
                    if new_deg == 1:
                        self.queue.append(other_pkt)

        return self.is_decoded()

    def _make_solved_packet(self, data: bytes, symbol: int) -> Packet:
        """Return a packet holding the solved *symbol* (subclasses use their packet type)."""
        return Packet(data, {symbol}, self.number_of_chunks, read_only=True)

    def _inactivation_solve(self, partial: bool = False) -> None:
        """Solve symbols the peeling could not resolve via a residual GEPP.

        After peeling stalls, every received packet has been XOR-reduced by the
        symbols solved so far, except for solved symbols it still lists (they are
        XORed out of a copy here).  A GEPP system over the *unsolved columns only*
        recovers the remaining symbols (RFC 5053 "inactivation decoding") in
        O(r·u²) instead of O(r·K²).  The provenance of every symbol solved here is
        the XOR of the provenances of the residual rows that GEPP combined into it.
        """
        unsolved = [i for i in range(self.number_of_chunks) if i not in self.solved_symbols]
        if not unsolved:
            return
        col_of = {sym: c for c, sym in enumerate(unsolved)}
        residual_rows: List[np.ndarray] = []
        residual_data: List[np.ndarray] = []
        residual_prov: List[int] = []
        for pkt in self.packets:
            used = pkt.get_used_packets()
            if not used:
                continue
            cols = [col_of[u] for u in used if u in col_of]
            if not cols:
                continue
            mask = np.zeros(len(unsolved), dtype=bool)
            mask[cols] = True
            d_arr = np.frombuffer(bytes(pkt.get_data()), dtype=np.uint8).copy()
            prov = getattr(pkt, "provenance", 0)
            for u in used:
                if u in self.solved_symbols and u not in col_of:
                    s_obj = self.solved_symbols[u]
                    s_arr = np.frombuffer(bytes(s_obj.get_data()), dtype=np.uint8)
                    n = min(len(d_arr), len(s_arr))
                    if n:
                        d_arr[:n] ^= s_arr[:n]
                    prov ^= getattr(s_obj, "provenance", 0)
            residual_rows.append(mask)
            residual_data.append(d_arr)
            residual_prov.append(prov)
        if not residual_rows or len({len(d) for d in residual_data}) != 1:
            return
        rgepp = GEPP(np.array(residual_rows, dtype=bool), np.array(residual_data, dtype=np.uint8))
        try:
            rgepp.solve(partial=True)
        except Exception:
            return
        res_map = rgepp.result_mapping
        for c, sym in enumerate(unsolved):
            if sym in self.solved_symbols or c >= len(res_map):
                continue
            row_idx = int(np.asarray(res_map[c]).reshape(-1)[0])
            if not 0 <= row_idx < len(rgepp.b):
                continue
            pkt = self._make_solved_packet(bytes(rgepp.b[row_idx]), sym)
            # residual rows combined into this symbol (rows beyond the received
            # ones are zero rows added for the partial solve)
            prov = 0
            for r in np.flatnonzero(rgepp.chunk_to_used_packets[row_idx]):
                if r < len(residual_prov):
                    prov ^= residual_prov[r]
            pkt.provenance = prov
            self._store_solved_symbol(sym, pkt)

    def _store_solved_symbol(self, sym_id: int, packet: Packet) -> None:
        """Record a solved symbol (and the header chunk, for symbol 0)."""
        self.solved_symbols[sym_id] = packet
        if sym_id == 0 and self.use_headerchunk and self.headerChunk is None:
            try:
                last_chunk_fmt = "I"
                if getattr(self, "config_map", None) is not None:
                    last_chunk_fmt = str(self.config_map.get("last_chunk_len_str", "I"))
                checksum_fmt = getattr(self, "checksum_len_str", "") or ""
                self.headerChunk = HeaderChunk(
                    packet,
                    last_chunk_len_format=last_chunk_fmt,
                    checksum_len_format=checksum_fmt,
                )
            except Exception:
                pass
        if sym_id != 0 or not self.use_headerchunk:
            self.decodedPackets.add(packet)

    def solve(self, partial: bool = False) -> bool:
        """Peel, then solve what peeling could not resolve by inactivation decoding."""
        self._peel()
        if not self.is_decoded():
            self._inactivation_solve(partial=partial)
            self._peel()
        return self.is_decoded()

    def getSolvedChunkIds(self) -> Set[int]:
        """Return the ids of the solved chunks."""
        return {sym for sym in self.solved_symbols if sym < self.number_of_chunks}

    def getSolvedCount(self) -> int:
        return len(self.getSolvedChunkIds())

    def solved_packets_in_order(self) -> List[Packet]:
        """Return the solved chunks as packets, ordered by chunk id."""
        return [self.solved_symbols[sym] for sym in sorted(self.getSolvedChunkIds())]

    def removeAndXorAuxPackets(
        self, packet: Union[Packet, RU10Packet, OnlinePacket]
    ) -> Union[List[bool], Set[int]]:
        return []

    def is_decoded(self) -> bool:
        return self.getSolvedCount() >= self.number_of_chunks

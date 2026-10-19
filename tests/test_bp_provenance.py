"""Packet provenance of the BP decoder (``GEPP.chunk_to_used_packets``).

The BP decoder records which received packets were XORed into every chunk, during
peeling (``Packet.xor_and_remove_packet``) and inactivation decoding
(``RU10BPDecoder._inactivation_solve``), and provides it as
``GEPP.chunk_to_used_packets`` with the same meaning as the GEPP decoder: row ``i``
lists the received packets (in input order) whose XOR is chunk ``i``.
"""

import copy

import numpy as np
import pytest
from norec4dna import Encoder, RU10BPDecoder, RU10Decoder, RU10Encoder
from norec4dna.distributions.RaptorDistribution import RaptorDistribution
from norec4dna.Packet import Packet

CHUNK_SIZE = 40


def _encode(tmp_path, size=4000, overhead=0.6, seed=7):
    """Encode random data; return (encoder, packets sorted by seed)."""
    path = tmp_path / "data.bin"
    path.write_bytes(np.random.default_rng(1).integers(0, 256, size, dtype=np.uint8).tobytes())
    n = Encoder.get_number_of_chunks_for_file_with_chunk_size(str(path), CHUNK_SIZE)
    encoder = RU10Encoder(
        str(path),
        n,
        RaptorDistribution(n),
        insert_header=True,
        id_len_format="H",
        number_of_chunks_len_format="H",
        seed=seed,
    )
    encoder.set_overhead_limit(overhead)
    encoder.encode_to_packets()
    return encoder, sorted(encoder.encodedPackets, key=lambda p: p.id)


def _copy(packet):
    """Independent copy of a packet (the decoders reduce packets in place)."""
    dist = getattr(packet, "dist", None)
    return copy.deepcopy(packet, {id(dist): dist} if dist is not None else None)


def _decode(decoder, packets, corrupt=None):
    """Feed copies of *packets*; return the received payloads in input order."""
    received = []
    for index, packet in enumerate(packets):
        packet = _copy(packet)
        if index == corrupt:
            data = np.frombuffer(bytes(packet.get_data()), dtype=np.uint8).copy()
            data[5] ^= 0xA5
            packet.set_data(data)
        received.append(np.frombuffer(bytes(packet.get_data()), dtype=np.uint8).copy())
        decoder.input_new_packet(packet)
    decoder.solve(partial=True)
    return received


def _truth(encoder, cid):
    """The encoder's chunk *cid* as a uint8 array."""
    chunk = encoder.chunks[cid]
    if isinstance(chunk, (bytes, bytearray)):
        return np.frombuffer(bytes(chunk), dtype=np.uint8)
    return np.asarray(chunk, dtype=np.uint8)


def _solved_rows(decoder):
    rows = np.asarray(decoder.GEPP.result_mapping).reshape(-1)
    return {cid: int(row) for cid, row in enumerate(rows) if int(row) >= 0}


def _chunk(decoder, row):
    return np.asarray(decoder.GEPP.b[row], dtype=np.uint8)


def _assert_provenance(decoder, received):
    """Every solved chunk equals the XOR of the received packets of its row."""
    inverse = decoder.GEPP.chunk_to_used_packets
    rows = _solved_rows(decoder)
    for cid, row in rows.items():
        acc = np.zeros_like(received[0])
        for packet in np.flatnonzero(inverse[row, : len(received)]):
            acc ^= received[packet]
        assert np.array_equal(acc, _chunk(decoder, row)), f"chunk {cid}"
        assert not inverse[row, len(received) :].any()
    return rows


def test_xor_and_remove_packet_tracks_provenance():
    """XORing a packet into another also XORs its provenance."""
    a = Packet(b"\x01\x02", {0, 1}, 3)
    b = Packet(b"\x03\x00", {1}, 3)
    a.provenance, b.provenance = 0b011, 0b110
    a.xor_and_remove_packet(b)
    assert a.get_used_packets() == {0}
    assert bytes(a.get_data()) == b"\x02\x02"
    assert a.provenance == 0b101
    assert Packet(b"\x00", {0}, 1).provenance == 0  # new packets have no history


@pytest.mark.parametrize("decoder_cls", [RU10BPDecoder, RU10Decoder])
def test_provenance_of_a_full_decode(tmp_path, decoder_cls):
    """Both decoders: correct chunks, and each chunk is the XOR of its packets."""
    encoder, packets = _encode(tmp_path)
    decoder = decoder_cls()
    received = _decode(decoder, packets)
    assert decoder.is_decoded()
    rows = _assert_provenance(decoder, received)
    assert len(rows) == encoder.number_of_chunks
    for cid, row in rows.items():
        assert np.array_equal(_chunk(decoder, row), _truth(encoder, cid))


def test_provenance_through_inactivation(tmp_path):
    """Symbols solved by inactivation decoding get the provenance of their rows."""
    calls = []
    original = RU10BPDecoder._inactivation_solve

    def spy(self, partial=False):
        before = len(self.solved_symbols)
        original(self, partial)
        calls.append(len(self.solved_symbols) - before)

    encoder, packets = _encode(tmp_path, overhead=0.4)
    for loss in (0.1, 0.15, 0.2, 0.25):  # until peeling stalls but the rank suffices
        keep = np.random.default_rng(3).random(len(packets)) >= loss
        decoder = RU10BPDecoder()
        decoder._inactivation_solve = spy.__get__(decoder)
        calls.clear()
        received = _decode(decoder, [p for p, k in zip(packets, keep) if k])
        if decoder.is_decoded() and sum(calls) > 0:
            break
    else:
        pytest.fail("no case in which inactivation decoding solved symbols")
    rows = _assert_provenance(decoder, received)
    for cid, row in rows.items():
        assert np.array_equal(_chunk(decoder, row), _truth(encoder, cid))


@pytest.mark.parametrize("decoder_cls", [RU10BPDecoder, RU10Decoder])
def test_provenance_of_a_partial_decode(tmp_path, decoder_cls):
    """With too few packets, the solved chunks still carry correct provenance."""
    _, packets = _encode(tmp_path)
    decoder = decoder_cls()
    received = _decode(decoder, packets[: int(0.7 * len(packets) / 1.6)])
    assert not decoder.is_decoded()
    _assert_provenance(decoder, received)


def test_received_system(tmp_path):
    """``get_received_system`` returns the equations in input order, before reduction."""
    _, packets = _encode(tmp_path)
    decoder = RU10BPDecoder()
    received = _decode(decoder, packets)
    a, b = decoder.get_received_system()
    assert a.shape == (len(received), decoder.number_of_chunks)
    assert np.array_equal(b, np.array(received))
    for row, packet in enumerate(packets):
        assert np.array_equal(a[row], decoder.removeAndXorAuxPackets(_copy(packet)))


@pytest.mark.parametrize("decoder_cls", [RU10BPDecoder, RU10Decoder])
def test_corrupt_packet_affects_exactly_its_chunks(tmp_path, decoder_cls):
    """A corrupt packet corrupts exactly the chunks whose provenance contains it."""
    encoder, packets = _encode(tmp_path)
    clean = decoder_cls()
    _decode(clean, packets)
    used = clean.GEPP.chunk_to_used_packets[list(_solved_rows(clean).values())].any(axis=0)
    victim = int(np.flatnonzero(used)[0])
    decoder = decoder_cls()
    _decode(decoder, packets, corrupt=victim)
    rows = _solved_rows(decoder)
    wrong = {
        cid
        for cid, row in rows.items()
        if not np.array_equal(_chunk(decoder, row), _truth(encoder, cid))
    }
    attributed = {
        cid for cid, row in rows.items() if decoder.GEPP.chunk_to_used_packets[row, victim]
    }
    assert wrong and wrong == attributed


def _lt_packets(tmp_path):
    from norec4dna.distributions.RobustSolitonDistribution import RobustSolitonDistribution
    from norec4dna.LTEncoder import LTEncoder

    path = tmp_path / "lt.bin"
    path.write_bytes(np.random.default_rng(2).integers(0, 256, 20000, dtype=np.uint8).tobytes())
    n = Encoder.get_number_of_chunks_for_file_with_chunk_size(str(path), 200)
    encoder = LTEncoder(
        str(path),
        n,
        RobustSolitonDistribution(S=n, delta=0.2, seed=2),
        chunk_size=200,
        insert_header=True,
        number_of_chunks_len_format="H",
        id_len_format="H",
    )
    encoder.set_overhead_limit(1.0)
    encoder.encode_to_packets()
    return encoder, sorted(encoder.encodedPackets, key=lambda p: p.id)


def _online_packets(tmp_path):
    from norec4dna.distributions.OnlineDistribution import OnlineDistribution
    from norec4dna.OnlineEncoder import OnlineEncoder

    path = tmp_path / "online.bin"
    path.write_bytes(np.random.default_rng(3).integers(0, 256, 72000, dtype=np.uint8).tobytes())
    n = Encoder.get_number_of_chunks_for_file_with_chunk_size(str(path), 200)
    encoder = OnlineEncoder(
        str(path),
        n,
        OnlineDistribution(0.07),
        0.07,
        5,
        number_of_chunks_len_format="H",
        check_block_number_len_format="H",
        quality_len_format="B",
        insert_header=True,
    )
    encoder.set_overhead_limit(1.0)
    encoder.encode_to_packets()
    return encoder, sorted(encoder.encodedPackets, key=lambda p: p.id)


@pytest.mark.parametrize("code", ["lt", "online"])
def test_shared_peeling_for_lt_and_online(tmp_path, code):
    """LT and Online BP decoders use the same peeling base, including provenance."""
    from norec4dna.LTBPDecoder import LTBPDecoder
    from norec4dna.OnlineBPDecoder import OnlineBPDecoder

    if code == "lt":
        encoder, packets = _lt_packets(tmp_path)
        decoder = LTBPDecoder()
    else:
        encoder, packets = _online_packets(tmp_path)
        decoder = OnlineBPDecoder()
    received = [np.frombuffer(bytes(p.get_data()), dtype=np.uint8).copy() for p in packets]
    for packet in packets:
        decoder.input_new_packet(_copy(packet))
    decoder.solve()
    assert decoder.is_decoded()
    assert decoder.getSolvedCount() == decoder.number_of_chunks
    matrix = decoder.provenance_matrix()
    for cid, packet in enumerate(decoder.solved_packets_in_order()):
        data = np.frombuffer(bytes(packet.get_data()), dtype=np.uint8)
        assert np.array_equal(data, _truth(encoder, cid)), f"chunk {cid}"
        acc = np.zeros_like(received[0])
        for p in np.flatnonzero(matrix[cid, : len(received)]):
            acc ^= received[p]
        assert np.array_equal(acc, data), f"provenance of chunk {cid}"
    a, _ = decoder.get_received_system()
    assert a.shape == (len(received), decoder.number_of_chunks)

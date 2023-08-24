import zstandard as zstd

file = "lichess_db_standard_rated_2013-01.pgn.zst"

with open(file, 'rb') as compressed:
    dctx = zstd.ZstdDecompressor()
    with dctx.stream_reader(compressed) as reader, open(file[0:-4], 'wb') as decompressed:
        while True:
            chunk = reader.read(16384)  # 16K chunks
            if not chunk:
                break
            decompressed.write(chunk)

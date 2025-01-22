import os
import gzip
import orjson
from orjson import JSONDecodeError
from pathlib import Path


class JsonlWriterV0:
    """
    Writes JSON lines to a file, optionally compressed with gzip.
    Batches of data can be written at once using write_batch().
    """

    def __init__(self, output_filename: str, compression: str = "gzip"):
        self.output_filename = output_filename
        os.makedirs(
            os.path.dirname(output_filename), exist_ok=True
        )  # ensures subdirs exist
        self.compression = compression
        self.file = None

    def __enter__(self):
        if self.compression == "gzip":
            self.file = gzip.open(self.output_filename, "wb")
        else:
            self.file = open(self.output_filename, "wb")
        return self

    def write_batch(self, batch):
        """
        Writes out each item in the given batch as a separate JSON line.
        """
        for doc in batch:
            line = orjson.dumps(doc, option=orjson.OPT_APPEND_NEWLINE)
            self.file.write(line)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()


class JsonlWriter:
    """
    Writes JSON lines to a (gzip-compressed) file by first writing to a
    temporary file named "<original>_tmp.jsonl.gz" and then renaming it to
    "<original>.jsonl.gz" upon exit.
    Batches of data can be written at once using write_batch().
    """

    def __init__(self, output_filename: str, compression: str = "gzip"):
        self.output_filename = Path(output_filename)
        # Ensure parent directories exist.
        os.makedirs(self.output_filename.parent, exist_ok=True)

        # Create the temporary filename by replacing the final ".jsonl.gz"
        # with "_tmp.jsonl.gz". This assumes output_filename ends with ".jsonl.gz".
        self.tmp_filename = Path(
            str(self.output_filename).replace(".jsonl.gz", "_tmp.jsonl.gz")
        )

        self.compression = compression
        self.file = None

    def __enter__(self):
        if self.compression == "gzip":
            self.file = gzip.open(self.tmp_filename, "wb")
        else:
            self.file = open(self.tmp_filename, "wb")
        return self

    def write_batch(self, batch):
        """
        Writes out each item in the given batch as a separate JSON line
        to the temporary file.
        """
        for doc in batch:
            line = orjson.dumps(doc, option=orjson.OPT_APPEND_NEWLINE)
            self.file.write(line)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()
        # Rename the temporary file to the final output name.
        self.tmp_filename.rename(self.output_filename)


class JsonlReader:
    """
    Reads JSON lines from a file, optionally compressed with gzip, yielding one
    batch at a time in an iterator. The batch_size is for buffering and can
    be adjusted for performance.
    """

    def __init__(self, file_name: str, compression: str = "gzip", batch_size: int = 64):
        self.file_name = file_name
        self.compression = compression
        self.batch_size = batch_size
        self.file = None

    def __enter__(self):
        if self.compression == "gzip":
            self.file = gzip.open(self.file_name, "rb")
        else:
            self.file = open(self.file_name, "rb")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()

    def __iter__(self):
        """
        Yields documents in batches
        """
        buffer = []
        for line in self.file:
            try:
                document = orjson.loads(line)
            except (EOFError, JSONDecodeError):
                continue
            buffer.append(document)
            if len(buffer) >= self.batch_size:
                yield buffer
                buffer.clear()

        # Yield any leftover documents in the buffer.
        if len(buffer) > 0:
            yield buffer
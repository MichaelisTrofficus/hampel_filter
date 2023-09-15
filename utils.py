import tarfile
import os.path


def make_tarfile(output_filename, source_dir):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))


if __name__ == "__main__":
    make_tarfile("dist/hampel-1.0.3.tar.gz", "dist/hampel-1.0.3")

import os
import shutil
import zmq.auth

from script.utils import fpath, fmdir


def generate_certificates():
    if os.path.exists(fpath("keys_server")):
        shutil.rmtree(fpath("keys_server"))
    if os.path.exists(fpath("keys_client")):
        shutil.rmtree(fpath("keys_client"))
    pubkcd = fmdir("keys_server", "public")
    secksd = fmdir("keys_server", "private")
    pubksd = fmdir("keys_client", "public")
    seckcd = fmdir("keys_client", "private")

    pubks, secks = zmq.auth.create_certificates(fpath("keys_server"), "server")
    pubkc, seckc = zmq.auth.create_certificates(fpath("keys_client"), "client")

    shutil.move(pubks, pubksd)
    shutil.move(pubkc, pubkcd)
    shutil.move(secks, secksd)
    shutil.move(seckc, seckcd)


if __name__ == "__main__":
    generate_certificates()

from script.client_core import Client
import argparse
import torch

torch.manual_seed(0)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-ip", "--ip", default="localhost", type=str,
                        help="master node ip (localhost)",
                        dest="ip")
    parser.add_argument("-p", "--port", default="51821", type=str,
                        help="master node port",
                        dest="port")

    parser.add_argument("-s", "--silent", action="store_true", default=False,
                        help="silent mode",
                        dest="s")

    parser.add_argument("-cuda", "--cuda", action="store_true", default=False,
                        help="use cuda",
                        dest="cuda")

    args = parser.parse_args()

    entity = Client(args)
    entity.run()


if __name__ == "__main__":
    main()

import argparse


if __name__=="__main__":
    # Used to pass agruments from the command line
    args=argparse.ArgumentParser()
    args.add_argument("--name","-n", default="Tanmay", type = str)
    args.add_argument("--age","-a", default=25, type = float)
    parse_args=args.parse_args()

    print(parse_args.name,"  ", parse_args.age)
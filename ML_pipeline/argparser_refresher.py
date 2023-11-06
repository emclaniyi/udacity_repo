import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime) - 15s %(message)s")
logger = logging.getLogger()

def go(args):

    logger.debug('this is a debug message')
    logger.info("This is a message")
    logger.warning("this is a warning")
    logger.error('this is an error')

    logger.info(f"This is {args.artifact_name}")
    logger.info(f"This is {args.optional_arg}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Use an artifact from W&B")

    parser.add_argument("--artifact_name", type=str, help="Name and version of W&B artifact", required=True)

    parser.add_argument("--optional_arg", type=float, help="an optional argument", required=False, default=2.3)

    args = parser.parse_args()

    go(args)
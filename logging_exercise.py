import logging

logging.basicConfig(
    filename='./results.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s'
)

def divide_vals(numerator, denominator):
    '''
    Args:
        numerator: (float) numerator of fraction
        denominator: (float) denominator of fraction

    Returns:
        fraction_val: (float) numerator/denominator
    '''
    try:
        assert denominator != 0
        assert isinstance(numerator, float)
        assert isinstance(denominator, float)
        fraction_val = numerator/denominator
        logging.info("SUCCESS: Denomainator is not zero and both values are both floats")
        
        return fraction_val
    except AssertionError:
        logging.error("ERROR: denominator cannot be zero")


def num_words(text):
    '''
    Args:
        text: (string) string of words

    Returns:
        num_words: (int) number of words in the string
    '''
    try:
        assert isinstance(text, str)
        num_words = len(text.split())
        logging.info('SUCCESS: text is a type of string')
        return num_words
    except AttributeError:
        logging.error("ERROR: text argument must be a string")

if __name__ == "__main__":
    divide_vals(3.4, 0)
    divide_vals(4.5, 2.7)
    divide_vals(-3.8, 2.1)
    divide_vals(1, 2)
    num_words(5)
    num_words('This is the best string')
    num_words('one')


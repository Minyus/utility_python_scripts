def input_shape_if_first(input_shape, first=False):
    if first:
        return input_shape
    return tuple([None] * len(input_shape))

if __name__ == "__main__":
    print('if first: ', input_shape_if_first((28,28,3),True))
    print('if not first: ', input_shape_if_first((28,28,3),False))
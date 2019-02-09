def str_list(input_list):
    return ['{}'.format(e) for e in input_list]

if __name__ == "__main__":
    input_list = ['a', 0]
    print(str_list(input_list))
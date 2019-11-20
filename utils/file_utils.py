def file_line_count(file_name):
    return sum(1 for line in open(file_name))

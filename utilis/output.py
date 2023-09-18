import os


def write_output(output_dir, local_rank, mess):
    if not isinstance(mess, str):
        mess = str(mess)
    output_file = os.path.join(output_dir, 'output_rank%d' % local_rank)
    with open(output_file, 'a') as file:
        file.write(mess)
        file.write('\n')

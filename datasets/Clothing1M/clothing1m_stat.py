from base_code.base_stat import StatRecorder

data_list = ['clean_label_kv.txt', 'noisy_label_kv.txt']
data_type = ['Multi-class', 'Multi-class']
column_name = ['Filename', 'Classes']

if __name__ == "__main__":
    clothing1mSR = StatRecorder(data_list=data_list, data_type=data_type, dataset_name="Clothing1M",
                                column_name=column_name, sep_type=' ')
    clothing1mSR.make_stat()

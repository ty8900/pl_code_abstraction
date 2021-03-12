import openpyxl
import pandas as pd
import os

"""
data_type 설명 (추가 예정)
'None': 필요 없는 stat
'Length': 데이터셋 크기만 저장
'Specific': 너무 구체적이라 df만 따로 리턴, 따로 핸들링
'Multi-label': 멀티라벨 데이터, 크기 및 분포 저장
'Multi-class': 멀티클래스 데이터, 크기 및 분포 저장
'Multi-label/Multi-class': 멀티라벨 + 멀티클래스 데이터, 크기 및 분포 저장
"""


class StatRecorder:
    def __init__(self, data_list, data_type,
                 dataset_name, column_name=None,
                 sep_type='\t',
                 stat_path: str = 'manual/auto_stat.xlsx',
                 spec_path: str = 'manual/spec_'):
        self.data_list = data_list
        self.data_type = data_type
        self.dataset_name = dataset_name
        self.column_name = column_name
        self.sep_type = sep_type
        self.stat_path = stat_path
        self.specific_path = spec_path

        if not os.path.exists(self.stat_path):
            stat_wb = openpyxl.Workbook()
            stat_wb.save(self.stat_path)

        self.df = pd.DataFrame
        self.length_df = pd.DataFrame(columns=['Name', 'Length'])

    def get_new_df(self, file_name: str, file_type: str):
        print("try to get new file: " + file_name + " ....")
        file_name = 'dataset/' + file_name
        if file_type == 'json':
            self.df = pd.read_json(file_name)
        elif file_type == 'txt':
            self.df = pd.read_csv(file_name, sep=self.sep_type, engine='python',
                                  names=self.column_name)
            if len(self.column_name) == 1:
                # only filename (filename have label info)
                # 특이 케이스인지, 흔한지에 따라 결정될듯. 일단 acs 한정한 코드
                self.df['Classes'] = self.df.Filename.str.split('/').str[0]
        print("Done!")

    def save_length(self):
        with pd.ExcelWriter(self.stat_path, mode='a') as writer:
            self.length_df.to_excel(writer, sheet_name="Length",
                                    index=False)

    def save_count(self, df, index):
        df['mean'] = pd.DataFrame(data=[df['Count'].mean()],
                                  index=[0])
        df['std'] = pd.DataFrame(data=[df['Count'].std()],
                                 index=[0])
        with pd.ExcelWriter(self.stat_path, mode='a') as writer:
            df.to_excel(writer, sheet_name=self.data_list[index],
                        index=False)

    def save_specific(self, index):
        self.df.to_csv(self.specific_path + self.data_list[index] + ".csv")

    def make_stat(self):
        for i in range(len(self.data_list)):
            cur_type = self.data_type[i]
            if cur_type == 'None':
                print("Current Data does not need to record")
                continue
            cur_data = self.data_list[i]
            self.get_new_df(cur_data, cur_data.split('.')[-1])

            self.length_df = self.length_df.append(
                pd.DataFrame({'Name': self.data_list[i], 'Length': len(self.df)},
                             index=[i]))
            if cur_type == 'Length':
                print("Successfully recorded length")
            elif cur_type == 'Specific':
                self.save_specific(i)
                print("Recorded specific df")
            elif cur_type == 'Multi-label':
                new_df = pd.DataFrame(columns=['Labels', 'Count'])
                for c in self.df.columns:
                    cs = self.df[c].value_counts()
                    if cs.count() != 2:
                        # 0 혹은 1만 나옴. 혹은 잘못된 칼럼(filename 같이)
                        if cs.keys()[0] == 0:
                            new_df = new_df.append(
                                pd.DataFrame({'Labels': [c], 'Count': [0]}))
                        elif cs.keys()[0] == 1:
                            new_df = new_df.append(
                                pd.DataFrame({'Labels': [c],
                                              'Count': [self.df[c].value_counts().values[0]]}))
                        else:
                            continue
                    else:
                        new_df = new_df.append(
                            pd.DataFrame({'Labels': [c],
                                          'Count': [self.df[c].value_counts()[1]]}))

                new_df = new_df.reset_index(drop=True)
                self.save_count(new_df, i)
                print("Successfully recorded multi-label stat")
            elif cur_type == 'Multi-class':
                cs = self.df['Classes'].value_counts()
                new_df = pd.DataFrame({'Classes': cs.index, 'Count': cs.values})
                self.save_count(new_df, i)
                print("Successfully recorded multi-class stat")
            elif cur_type == 'Multi-label/Multi-class':
                new_df = pd.DataFrame(columns=['Labels', 'Classes', 'Count'])
                for c in self.df.columns:
                    if c == 'Filename':
                        continue
                    cs = self.df[c].value_counts()
                    new_df = new_df.append(pd.DataFrame({'Labels': c,
                                                         'Classes': cs.index,
                                                         'Count': cs.values}))
                new_df = new_df.reset_index(drop=True)
                self.save_count(new_df, i)
                print("Successfully recorded multi-label/multi-class stat")

        self.save_length()
        print("Statistics making ended.")

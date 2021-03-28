import openpyxl
import pandas as pd
import os
from scipy import io

"""
data_type 설명 (추가 예정)
'None': 필요 없는 stat
'Length': 데이터셋 크기만 저장
'Specific': 너무 구체적이라 df만 따로 리턴, 따로 핸들링
'Multi-label': 멀티라벨 데이터, 크기 및 분포 저장
'Multi-class': 멀티클래스 데이터, 크기 및 분포 저장
'Multi-label/Multi-class': 멀티라벨 + 멀티클래스 데이터, 크기 및 분포 저장
"""

"""
<How to use>
데이터 분포 추출 코드.
(args)
data_list : 추출한 파일명 리스트
data_type : 파일들의 type. .txt, .json, .mat 등등
dataset_name : 데이터셋 이름. (sheet이름, path 설정 등)
column_name : dataframe에 쓸 이름. txt등의 경우 이름이 없을 수 있어 설정 시 사용
sep_type : dataframe deliminator. default '\t'이나 ' '일때도 있었음.
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

        # check directory manual exists
        if not os.path.exists('./manual'):
            os.mkdir('./manual')

        # check stat file exists
        if not os.path.exists(self.stat_path):
            stat_wb = openpyxl.Workbook()
            stat_wb.save(self.stat_path)

        # for dataset
        self.df = pd.DataFrame
        # for record length
        self.length_df = pd.DataFrame(columns=['Name', 'Length'])

    # get new dataframe from file_name.
    # may need to override for specific case; e.g. .mat file.
    # (파일에서 dataframe을 읽는 과정, 좀 특이한 경우가 많아 그 경우 override하여 사용)
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
        elif file_type == 'mat':
            dic = io.loadmat(file_name)
            self.df = pd.DataFrame.from_dict(dic, columns=self.column_name)
        print("Done!")

    # 데이터 길이(크기) plot.
    def save_length(self):
        with pd.ExcelWriter(self.stat_path, mode='a') as writer:
            self.length_df.to_excel(writer, sheet_name="Length",
                                    index=False)

    # 데이터 통계량 (평균, 표준편차) plot.
    def save_count(self, df, index):
        df['mean'] = pd.DataFrame(data=[df['Count'].mean()],
                                  index=[0])
        df['std'] = pd.DataFrame(data=[df['Count'].std()],
                                 index=[0])
        with pd.ExcelWriter(self.stat_path, mode='a') as writer:
            sheet_name = self.data_list[index]
            if '/' in self.data_list[index]:
                sheet_name = self.data_list[index].split('/')[-1]
            df.to_excel(writer, sheet_name=sheet_name,
                        index=False)

    # 너무 핸들링하기 어려운 데이터 분포는 따로 데이터프레임만 저장.
    def save_specific(self, index):
        self.df.to_csv(self.specific_path + self.data_list[index] + ".csv")

    # 데이터 분석 main 코드
    def make_stat(self):
        for i in range(len(self.data_list)):
            cur_type = self.data_type[i]
            # type None check
            if cur_type == 'None':
                print("Current Data does not need to record")
                continue

            # get new dataframe from file
            # length_df is for record data's length
            cur_data = self.data_list[i]
            self.get_new_df(cur_data, cur_data.split('.')[-1])
            self.length_df = self.length_df.append(
                pd.DataFrame({'Name': self.data_list[i], 'Length': len(self.df)},
                             index=[i]))
            self.save_length()
            # (1) 길이만 저장하는 경우.
            if cur_type == 'Length':
                print("Successfully recorded length")
            # (2) 너무 복잡한 경우. df만 따로 저장.
            elif cur_type == 'Specific':
                self.save_specific(i)
                print("Recorded specific df")
            # (3) Multi-label / binary class 데이터.
            # 0또는 1로만 데이터 분포 분석.
            elif cur_type == 'Multi-label':
                new_df = pd.DataFrame(columns=['Labels', 'Count'])
                for c in self.df.columns:
                    cs = self.df[c].value_counts()
                    if cs.count() != 2:
                        # 0 혹은 1만 나와야함(keys = 2) exception handling.
                        if cs.keys()[0] == 0:
                            print("Column exception: there is not any key")
                            new_df = new_df.append(
                                pd.DataFrame({'Labels': [c], 'Count': [0]}))
                        elif cs.keys()[0] == 1:
                            print("Column exception: there is only one key")
                            new_df = new_df.append(
                                pd.DataFrame({'Labels': [c],
                                              'Count': [self.df[c].value_counts().values[0]]}))
                        else:
                            print("Column exception: there are non-binary keys")
                            continue
                    new_df = new_df.append(
                            pd.DataFrame({'Labels': [c],
                                          'Count': [self.df[c].value_counts()[1]]}))

                new_df = new_df.reset_index(drop=True)
                self.save_count(new_df, i)
                print("Successfully recorded multi-label stat")
            # (4) Multi-class 데이터. (한 label 종류)
            # 중요: 반드시 Classes 라는 이름의 column 필요. 이 기준으로 분석하기 때문.
            elif cur_type == 'Multi-class':
                if 'Classes' not in df.keys():
                    print("Multi-class exception: there is no Classes column.")
                    raise NameError
                cs = self.df['Classes'].value_counts()
                new_df = pd.DataFrame({'Classes': cs.index, 'Count': cs.values})
                self.save_count(new_df, i)
                print("Successfully recorded multi-class stat")
            # (5) Multi-label/Multi-class 데이터. (여러 종류, 여러 선택지)
            # column은 각 label에 대한 정보.
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

        print("Statistics making ended.")

# %%
import concurrent.futures

def process_data(data):
    # 데이터 처리 함수
    pass

if __name__ == '__main__':
    data_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(process_data, data_list)
    

# %%


# ADL HW2: m11203404 陳旭霖
以下內容包含:
* Build up environment
* Download models
* Preprocess and Inference
* Training 
* Contact Information

## Build up environment
- Step1: Please create and activate your venv(python=3.9) first
- Step2: install torch
    ```
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
    ```
- Step3: install requirement.txt 
    ```
    pip install -r requirement.txt
    ```

## Download models
- Option1: Download by Shell Scripts
    ```
    bash ./download.sh
    ```
- Option2: Download ADLHW2_beam_50k.zip by gDrive: 
    
    https://drive.google.com/file/d/1W-8NmQ3GYWYQrBQ1WEq-NrYjcHDwrQT5/view?usp=drive_link

## Preprocess and Inferance
- Execute by Shell Scripts
    ```
    bash ./run.sh /path/to/input.jsonl /path/to/output.jsonl
    ```
## Training
- Required jsonl format:
    ```
    {
        "title": "門票秒殺！YOASOBI開演唱會 台粉求票「願改名」",
        "maintext": "日本人氣天團「YOASOBI」明年1月21號來台開唱…….回應歌迷期待。",
        "id": "1"
    }
    ```
- Step1: 先將提供的jsonl檔案修改成程式可以讀取的.csv格式
    ```
    python modify_jsonl_to_csv.py -i  ./data/train.jsonl -o ./data/modify_train.csv

    python modify_jsonl_to_csv.py -i  ./data/public.jsonl -o ./data/modify_public.csv
    ```

- Step2: Fine tune
    ```
    # model save at ./tmp
    python train.py --train_file ./data/modify_train.csv --valid_file ./data/modify_public.csv 
    ```


## Contact
- Email: charles77778888asd@gmail.com 
- linkedin: www.linkedin.com/in/旭霖-陳-b34102277






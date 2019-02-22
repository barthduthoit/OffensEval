mkdir -p data/raw
mkdir -p data/test/task_a
mkdir data/test/task_b
mkdir data/test/task_c

echo Downloading archives...
wget -q --show-progress -cO - https://competitions.codalab.org/my/datasets/download/60e40c68-a85d-4320-bef1-d2fe26bb45ca > data/raw/start-kit.zip
wget -q --show-progress -cO - https://competitions.codalab.org/my/datasets/download/5cac0f56-bb6d-40fa-8041-caf8aa13d09d > data/test/task-a.zip
wget -q --show-progress -cO - https://competitions.codalab.org/my/datasets/download/bb373027-c8b7-48ab-9729-b1ab3fb51c17 > data/test/task-b.zip
wget -q --show-progress -cO - https://competitions.codalab.org/my/datasets/download/38273e56-2ab0-4773-82bf-95aec51bba69 > data/test/task-c.zip

echo Extracting archives...
unzip -qq data/raw/start-kit.zip -d data/raw/
unzip -qq data/raw/training-v1.zip -d data/raw/
unzip -qq data/raw/trial-data.zip -d data/raw/
unzip -qq data/test/task-a.zip -d data/test/task_a/
unzip -qq data/test/task-b.zip -d data/test/task_b/
unzip -qq data/test/task-c.zip -d data/test/task_c/

rm data/raw/*.zip
rm data/test/*.zip

mkdir data/additional/
echo Downloading additional datasets
wget -q --show-progress https://github.com/kmi-linguistics/trac-1/raw/master/english/agr_en_train.csv -P data/additional/
wget -q --show-progress https://github.com/t-davidson/hate-speech-and-offensive-language/raw/master/data/labeled_data.csv -P data/additional
wget -q --show-progress https://github.com/mammothb/symspellpy/raw/master/symspellpy/frequency_dictionary_en_82_765.txt -P data/

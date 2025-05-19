# MixATIS测试

echo "Testing all models on MixATIS dataset"

DATA_PATH="data/MixATIS_clean_processed.pth"
EPOCHS=100

python main.py --data_path "${DATA_PATH}" --model_type cnn --lr 0.001  --epochs "${EPOCHS}" --embed_dim 64 
python main.py --data_path "${DATA_PATH}" --model_type lstm --lr 0.001  --epochs "${EPOCHS}" --embed_dim 64 --hidden_dim 64
python main.py --data_path "${DATA_PATH}" --model_type gru --lr 0.0002  --epochs "${EPOCHS}" --embed_dim 64 --hidden_dim 128
python main.py --data_path "${DATA_PATH}" --model_type rbfn --lr 0.001  --epochs "${EPOCHS}" --embed_dim 64 --hidden_dim 64
python main.py --data_path "${DATA_PATH}" --model_type transformer --lr 0.0001 --epochs "${EPOCHS}" --d_model 256 --nhead 8 --num_layers 6
python main.py --data_path "${DATA_PATH}" --model_type bert --lr 0.0002 --epochs "${EPOCHS}" --d_model 512 --nhead 8 --num_layers 2
python main.py --data_path "${DATA_PATH}" --model_type bert-freeze --lr 0.001 --epochs "${EPOCHS}" 

echo "All models have been tested on MixATIS dataset"
echo "-"*50

# MixSNIPS测试

echo "Testing all models on MixSNIPS dataset"
DATA_PATH="data/MixSNIPS_clean_processed.pth"
EPOCHS=50

python main.py --data_path "${DATA_PATH}" --model_type cnn --lr 0.001  --epochs "${EPOCHS}" --embed_dim 64 
python main.py --data_path "${DATA_PATH}" --model_type lstm --lr 0.001  --epochs "${EPOCHS}" --embed_dim 32 --hidden_dim 128
python main.py --data_path "${DATA_PATH}" --model_type gru --lr 0.001 --epochs "${EPOCHS}" --embed_dim 64 --hidden_dim 128
python main.py --data_path "${DATA_PATH}" --model_type rbfn --lr 0.001  --epochs "${EPOCHS}" --embed_dim 64 --hidden_dim 128
python main.py --data_path "${DATA_PATH}" --model_type transformer --lr 0.0001 --epochs "${EPOCHS}" --d_model 512 --nhead 8 --num_layers 6
python main.py --data_path "${DATA_PATH}" --model_type bert --lr 0.0002 --epochs "${EPOCHS}" --d_model 256 --nhead 8 --num_layers 6
python main.py --data_path "${DATA_PATH}" --model_type bert-freeze --lr 0.001 --epochs "${EPOCHS}" 

echo "All models have been tested on MixSNIPS dataset"
echo "-"*50

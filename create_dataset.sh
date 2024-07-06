# Script for creating npy files with candidate slices and labels
# [GPS - 01/26/2019]

SIZE=${1:-32}
TESTING=${2:-False}
python src/create_object_slices.py \
		   --slice-size $SIZE \

python src/create_db.py \
                   --slice-size $SIZE \
                   --testing $TESTING

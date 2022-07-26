XINXIANG 测试过程：将测试图片放入测试文件夹中， 在终端输入python tools/infer/predict_system.py --image_dir="测试文件夹路径" --det_model_dir="./inferencejc/" --rec_model_dir="./inferenceshibie" --use_angle_cls=false --rec_char_dict_path=“xxocr.txt"  --det_db_unclip_ratio=2 --use_gpu=False
在inference_results文件夹里查看结果。

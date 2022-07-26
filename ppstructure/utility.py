
from PIL import Image
import numpy as np
from tools.infer.utility import draw_ocr_box_txt, init_args as infer_args


def init_args():
    parser = infer_args()

    # params for output
    parser.add_argument("--output", type=str, default='./output/table')
    # params for table structure
    parser.add_argument("--table_max_len", type=int, default=488)
    parser.add_argument("--table_model_dir", type=str)
    parser.add_argument("--table_char_type", type=str, default='en')
    parser.add_argument("--table_char_dict_path", type=str, default="../ppocr/utils/dict/table_structure_dict.txt")
    parser.add_argument("--layout_path_model", type=str, default="lp://PubLayNet/ppyolov2_r50vd_dcn_365e_publaynet/config")
    return parser


def parse_args():
    parser = init_args()
    return parser.parse_args()


def draw_structure_result(image, result, font_path):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    boxes, txts, scores = [], [], []
    for region in result:
        if region['type'] == 'Table':
            pass
        else:
            for box, rec_res in zip(region['res'][0], region['res'][1]):
                boxes.append(np.array(box).reshape(-1, 2))
                txts.append(rec_res[0])
                scores.append(rec_res[1])
    im_show = draw_ocr_box_txt(image, boxes, txts, scores, font_path=font_path,drop_score=0)
    return im_show
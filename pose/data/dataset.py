import orjson as json
from torch.utils.data import Dataset


class CocoDataset(Dataset):
    def __init__(self, anno_fname, image_dir):
        self.image_dir = image_dir
        anno = json.loads(open(anno_fname))

        # load images
        self.images = {}
        for _img in anno['images']:
            v = {}
            for att in ['file_name', 'height', 'width']:
                v[att] = _img[att]
            images[_img['id']] = v

        # load keypoints
        self.keypoints = {}
        for _anno in anno['annotations']:
            if _anno['iscrowd'] != 0:
                continue
            #TODO
            pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

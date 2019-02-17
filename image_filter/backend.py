from fastai.vision import *
from path import Path as path

##UNTESTED: Use at your own risk!!!

def predict(train_dir, unlabeled_dir, dest_dir, imgmod = "resnet34", bs = 64, \
        size = 224):
    data = ImageDataBunch.from_folder(train_dir, valid_pct = 0.2, size=size, \
                                      bs = bs, ds_tfms=get_transforms()).normalize(imagenet_stats)
    learn = create_cnn(data, models.__dict__[imgmod], metrics=error_rate)    
    learn.load("model")
    for f in path(unlabeled_dir).files():
        try:
            img = open_image(f)
            pred = str(learn.predict(img)[0])
            dest = path(dest_dir + "/" + pred)
            dest.mkdir_p()
            path(f).copy(dest)
        except Exception:
            print("Error in {}".format(str(f.name)))        
    
def predict_and_train(train_dir, unlabeled_dir, dest_dir, imgmod = "resnet34",\
                      bs = 64, size = 224, s1_epochs = 4, s2_epochs = 1):
    train(train_dir, unlabeled_dir, dest_dir, imgmod, bs, size, s1_epochs, s2_epochs)
    predict(train_dir, unlabeled_dir, dest_dir, imgmod, bs, size)
    


def train(train_dir, imgmod = "resnet34", bs = 64, size = 224, s1_epochs = 4, \
          s2_epochs = 1):
    data = ImageDataBunch.from_folder(train_dir, valid_pct = 0.2, size=size, \
                                      bs = bs, ds_tfms=get_transforms()).normalize(imagenet_stats)
    learn = create_cnn(data, models.__dict__[imgmod], metrics=error_rate)
    learn.fit_one_cycle(s1_epochs)
    if s2_epochs > 0:
        learn.unfreeze()
        learn.fit_one_cycle(s2_epochs)
    learn.save("model")
    
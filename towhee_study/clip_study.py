from towhee.models.clip import create_model
from torchvision import transforms
from towhee.dc2 import pipe, ops, DataCollection

model = create_model('clip_vit_b32', pretrained=True, device='cpu')


# def encode_text(x):
#     return model.encode_text(x, multilingual=True).squeeze(0).detach().cpu().numpy()


img_pipe = (
    pipe.input('url')
    .map('url', 'img', ops.image_decode.cv2_rgb())
    .map('img', 'vec', ops.image_text_embedding.clip(model_name='clip_vit_base_patch16', modality='image'))
    .output('img', 'vec')
)

text_pipe = (
    pipe.input('text')
    .map('text', 'vec', ops.image_text_embedding.clip(model_name='clip_vit_base_patch16', modality='text'))
    .output('text', 'vec')
)

# DataCollection(img_pipe('./teddy.jpg')).show()
# DataCollection(text_pipe('A teddybear on a skateboard in Times Square.')).show()


if __name__ == '__main__':
    # print(encode_text("text"))
    print(DataCollection(text_pipe('A teddybear on a skateboard in Times Square.'))[0]['vec'])

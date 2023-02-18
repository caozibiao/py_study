import towhee


if __name__ == '__main__':
    # create image embeddings and build index
    (
        towhee.glob['file_name']('./*.jpg')
        .image_decode['file_name', 'img']()
        .image_text_embedding.clip['img', 'vec'](model_name='clip_vit_b32', modality='image')
        .tensor_normalize['vec', 'vec']()
        .to_faiss[('file_name', 'vec')](findex='./index.bin')
    )

    # search image by text
    results = (
        towhee.dc['text'](['puppy Corgi'])
        .image_text_embedding.clip['text', 'vec'](model_name='clip_vit_b32', modality='text')
        .tensor_normalize['vec', 'vec']()
        .faiss_search['vec', 'results'](findex='./index.bin', k=3)
        .select['text', 'results']()
    )
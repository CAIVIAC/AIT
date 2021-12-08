import torch

from transformer.Models import Transformer

def adaptive_image_transformer():
    # >> input initialization
    batch_size = 4
    num_props = 128
    channels = 1024

    props_feat = torch.rand(batch_size*num_props, channels, 7, 7).cuda()
    non_qry = torch.rand(batch_size, channels, 8, 8).cuda()


    """
    Adaptive Image Transformer
    """

    # >> please add transformer in the __init__() of the class _fasterRCNN()
    AIT = Transformer(
        d_k=64,
        d_v=64,
        d_model=channels // 2,
        d_word_vec=channels // 2,
        d_inner=channels * 2,
        n_position= 8 * 8,
        n_layers=1,
        n_head=8,
        dropout=0.1)

    AIT = AIT.cuda() # use cuda

    # >> usage of AIT
    # Note that x_query should be zero-padded to size 8*8 
    out = AIT(x_props=props_feat, x_query=non_qry) # [batch_size*num_props, c=1024, h=8, w=8]

    # >> following lines show the structure of AIT as well as the input and output shapes
    print(AIT)

    print('-'*50)
    print('Input shape: \n\
        Proposal: {}\n  \
        Query : {}\n'.format(
        ' x '.join(list(map(str, props_feat.shape))),
        ' x '.join(list(map(str, non_qry.shape)))
    ))
    print('Output shape: \n\
        Reconstruted Proposal: {}'.format(
        ' x '.join(list(map(str, out.shape))),
    ))

if __name__ == "__main__":
    """ usage
    $ python adaptive_image_transformer.py
    """
    adaptive_image_transformer()

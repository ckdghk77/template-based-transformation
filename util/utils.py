import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import matplotlib
import cv2
from matplotlib import gridspec
#matplotlib.use('Agg')


def gen_onehot_tensor(num, class_idx, tot_class=10, gpu=False):

    target_label = torch.zeros(size=(num, tot_class));
    target_label[torch.arange(target_label.size(0)), class_idx] = 1.0

    if gpu :
        target_label = target_label.cuda()

    return target_label


def cosine_dist(feature1, feature2, is_normed=False) :


    l2norm_f1 = feature1 / (feature1.pow(2).sum(1, keepdim=True).sqrt() + 1e-8)
    l2norm_f2 = feature2 / (feature2.pow(2).sum(1, keepdim=True).sqrt() + 1e-8)

    cos_dist = (1 - F.cosine_similarity(l2norm_f1, l2norm_f2, dim=1)).mean();

    return cos_dist

def cosine_sim(feature1, feature2, is_normed=False) :

    l2norm_f1 = feature1 / (feature1.pow(2).sum(1, keepdim=True).sqrt() + 1e-8)
    l2norm_f2 = feature2 / (feature2.pow(2).sum(1, keepdim=True).sqrt() + 1e-8)

    cos_sim = F.cosine_similarity(l2norm_f1, l2norm_f2, dim=1).mean();

    return cos_sim

def nll_gaussian(preds, target, variance, weighting = None, add_const=False):
    neg_log_p = (((preds - target) ** 2) / (2 * variance ** 2))

    return torch.mean(neg_log_p)




def figure_to_array(fig):
    """
    plt.figure를 RGBA로 변환(layer가 4개)
    shape: height, width, layer
    """
    fig.canvas.draw()
    return np.array(fig.canvas.renderer._renderer)

def rgba2rgb( rgba, background=(255,255,255) ):
    row, col, ch = rgba.shape

    if ch == 3:
        return rgba

    assert ch == 4, 'RGBA image has 4 channels.'

    rgb = np.zeros( (row, col, 3), dtype='float32' )
    r, g, b, a = rgba[:,:,0], rgba[:,:,1], rgba[:,:,2], rgba[:,:,3]

    a = np.asarray( a, dtype='float32' ) / 255.0

    R, G, B = background

    rgb[:,:,0] = r * a + (1.0 - a) * R
    rgb[:,:,1] = g * a + (1.0 - a) * G
    rgb[:,:,2] = b * a + (1.0 - a) * B

    return np.asarray( rgb, dtype='uint8' )



def visualize_velocity_field(offset, pool_size=1) :

    assert ((offset.shape[-2]/pool_size == offset.shape[-2]//pool_size) or\
           (offset.shape[-1] / pool_size == offset.shape[-1]//pool_size)), "inappropriate pool_size";

    fig, axes = plt.subplots(1,1);


    offset_pooled = F.avg_pool2d(offset.unsqueeze(0), kernel_size=pool_size, stride=pool_size)[0];

    offset_pooled = offset_pooled.cpu().data.numpy();
    #offset_pooled[0,:,:] = -1*offset_pooled[0, :, :];
    offset_pooled[1,:,:] = -1*offset_pooled[1, :, :];

    x,y = np.meshgrid(np.arange(offset.shape[-2]//pool_size),np.arange(offset.shape[-1]//pool_size));

    axes.quiver(x,y,offset_pooled[1],offset_pooled[0]);
    axes.axis('off')

    plt.close('all')

    return rgba2rgb(figure_to_array(fig))


def draw_img(img) :

    fig, axes = plt.subplots(1, 1);

    fig.set_size_inches(2, 2);

    if len(img)== 3 :
        axes.imshow(np.transpose(img.data.cpu().numpy(), (1, 2, 0)));
    else :
        axes.imshow(img.data.cpu().numpy(), cmap='Greys')


    axes.axis('off')

    plt.close('all')

    return rgba2rgb(figure_to_array(fig))

def draw_attentions(datas, attentions, plt_save=False) :

    gs = gridspec.GridSpec(
        nrows=datas.shape[0], ncols= attentions.shape[1] + 1,
        left=0.125, bottom=0.11, right=0.9, top=0.88, wspace=0.2, hspace=0.2)

    fig = plt.figure();
    #fig.set_size_inches(4 * (attentions.shape[1]-1)/len(datas), 4)
    fig.set_size_inches(4, 4)

    # fig, axes = plt.subplots(datas.shape[0], transformed.shape[1] + 1);

    for i in range(attentions.shape[0]):

        for j in range(attentions.shape[1]):

            ax = fig.add_subplot(gs[i, j+1]);

            np_img = attentions[i, j].data.cpu().numpy();
            np_img = (np_img - np_img.min()) \
                     / (np_img.max() - np_img.min() + 1e-6);


            np_img = np_img * 255.0;

            heatmap = cv2.applyColorMap(np_img.astype(np.uint8), cv2.COLORMAP_JET);

            ax.imshow(heatmap[:,:,(2,1,0)])
            ax.axis('off')


        ax = fig.add_subplot(gs[i, 0]);


        ax.imshow(np.transpose(datas[i].data.cpu().numpy(), (1, 2, 0)));
        ax.axis('off')

    if plt_save:
        #plt.show()

        fig = plt.figure();
        plt.imshow(heatmap[:,:,(2,1,0)], cmap="jet");
        plt.colorbar()
        #plt.show()
        plt.savefig("fig/attention.png", transparent=True);


    plt.close('all')

    return rgba2rgb(figure_to_array(fig))


def draw_template(template, template_dim) :


    fig, axes = plt.subplots(1,template_dim);

    fig.set_size_inches(6.5, 2);

    for i in range(template_dim) :
        axes[i].imshow(np.squeeze(template[i].data.cpu().numpy()), cmap='Greys')
        axes[i].axis('off')


    '''
    fig, axes = plt.subplots(1, 1);

    fig.set_size_inches(2, 2);

    axes.imshow(np.transpose(template.data.cpu().numpy(), (1, 2, 0)));
    
    axes.axis('off')
    '''
    plt.close('all')

    return rgba2rgb(figure_to_array(fig))

def draw_velocity_array(velocity, data, tar_dat) :
    fig, axes = plt.subplots(1,3);
    velo_img = visualize_velocity_field(velocity[0]);

    axes[0].imshow(np.transpose(data[0].data.cpu().numpy(), (1, 2, 0)));
    axes[1].imshow(velo_img);
    axes[2].imshow(np.transpose(tar_dat[0].data.cpu().numpy(), (1, 2, 0)));

    axes[0].axis("off")
    axes[1].axis("off")
    axes[2].axis("off")
    plt.close('all')

    return rgba2rgb(figure_to_array(fig))


def draw_interp_array(transformed, datas, plt_save=False) :

    gs = gridspec.GridSpec(
        nrows=datas.shape[0], ncols= transformed.shape[1] + 1,
        left=0.125, bottom=0.11, right=0.9, top=0.88, wspace=0.2, hspace=0.2)

    fig = plt.figure();
    fig.set_size_inches(4, 4)
    #fig, axes = plt.subplots(datas.shape[0], transformed.shape[1] + 1);

    data_len = datas.shape[0];


    for i in range(transformed.shape[0]) :

        for j in range(transformed.shape[1]) :

            if data_len == 1:
                ax = fig.add_subplot(gs[j]);
            else :
                ax = fig.add_subplot(gs[i,j]);


            if datas.shape[1] == 3:
                ax.imshow(np.transpose(transformed[i,j].data.cpu().numpy(), (1,2,0)));
            else :
                ax[j].imshow(np.squeeze(transformed[i,j].data.cpu().numpy()), cmap='Greys')

            if data_len == 1 :
                ax.axis('off')
            else:
                ax.axis('off')

        if data_len == 1 :
            ax = fig.add_subplot(gs[j+1]);
        else :
            ax = fig.add_subplot(gs[i,j+1]);

        if datas.shape[1] == 3:
            ax.imshow(np.transpose(datas[i].data.cpu().numpy(), (1,2,0)));
        else :
            ax.imshow(np.squeeze(datas[i].data.cpu().numpy()), cmap='Greys')
        ax.axis('off')

    if plt_save :

        #plt.show()
        plt.savefig("fig/interpolated.png", transparent=True);

    plt.close('all')

    return rgba2rgb(figure_to_array(fig))

def draw_shuffle_array(transformed, datas, plt_save=False) :

    gs = gridspec.GridSpec(
        nrows=transformed.shape[0] + 1, ncols= transformed.shape[0] + 1,
        left=0.125, bottom=0.11, right=0.9, top=0.88, wspace=0.2, hspace=0.2)

    fig = plt.figure();
    fig.set_size_inches(4, 4)

    ax = fig.add_subplot(gs[0,0]);
    ax.axis('off');

    for i in range(transformed.shape[0]) : # fill value

        ax = fig.add_subplot(gs[0, i+1]);
        if datas.shape[1] == 3:
            ax.imshow(np.transpose(datas[i].data.cpu().numpy(), (1,2,0)));
        else :
            ax.imshow(np.squeeze(datas[i].data.cpu().numpy()), cmap='Greys')
        ax.axis('off')

    for i in range(transformed.shape[0]) : # fill spatial
        ax = fig.add_subplot(gs[i+1, 0]);
        if datas.shape[1] == 3:
            ax.imshow(np.transpose(datas[i].data.cpu().numpy(), (1,2,0)));
        else :
            ax.imshow(np.squeeze(datas[i].data.cpu().numpy()), cmap='Greys')
        ax.axis('off')


    for i in range(transformed.shape[0]) :
        for j in range(transformed.shape[0]) :
            ax = fig.add_subplot(gs[i+1, j+1]);

            if transformed[i,j].shape[0] == 3:
                ax.imshow(np.transpose(transformed[i,j].data.cpu().numpy(), (1,2,0)))
            else :
                ax.imshow(np.squeeze(transformed[i,j].data.cpu().numpy()), cmap='Greys')
            ax.axis('off')


    if plt_save :
        #plt.show()
        plt.savefig("fig/shuffled.png", transparent=True);

    plt.close('all')

    return rgba2rgb(figure_to_array(fig))
'''
def draw_shuffle_array(template, transformed, datas, template_dim) :

    fig, axes = plt.subplots(transformed.shape[0]+1, transformed.shape[0]+1);
    fig.set_size_inches(6.5, 6)
    #fig.set_size_inches(13, 12)

    colors = sns.color_palette("colorblind", template_dim);

    np_template = template.cpu().data.numpy();
    #np_spatial_transformed = spatial_transformed.cpu().data.numpy()

    color_mapped_template = np.zeros(shape=(template.shape[0], template.shape[1], 3));
    #color_mapped_spatial_template = np.zeros(shape=(spatial_transformed.shape[0], spatial_transformed.shape[1],
    #                                                template.shape[0], template.shape[1], 3));
    for idx, color in enumerate(colors):
        row, col = np.where(np_template == idx);
        color_mapped_template[row, col, :] = color;

        #idx0, idx1, row, col = np.where(np_spatial_transformed == idx);
        #color_mapped_spatial_template[idx0, idx1, row, col, :] = color;


    axes[0,0].axis('off');
    draw_target = color_mapped_template;
    axes[0,0].imshow(draw_target);


    for i in range(transformed.shape[0]) : # fill value
        if datas.shape[1] == 3:
            axes[0,i+1].imshow(np.transpose(datas[i].data.cpu().numpy(), (1,2,0)));
        else :
            axes[0,i+1].imshow(np.squeeze(datas[i].data.cpu().numpy()), cmap='Greys')
        axes[0,i+1].axis('off')

    for i in range(transformed.shape[0]) : # fill spatial
        if datas.shape[1] == 3:
            axes[i+1,0].imshow(np.transpose(datas[i].data.cpu().numpy(), (1,2,0)));
        else :
            axes[i+1,0].imshow(np.squeeze(datas[i].data.cpu().numpy()), cmap='Greys')
        axes[i+1, 0].axis('off')


    for i in range(transformed.shape[0]) :
        for j in range(transformed.shape[0]) :
            if transformed[i,j].shape[0] == 3:
                axes[i+1,j+1].imshow(np.transpose(transformed[i,j].data.cpu().numpy(), (1,2,0)))
            else :
                axes[i+1,j+1].imshow(np.squeeze(transformed[i,j].data.cpu().numpy()), cmap='Greys')

            axes[i+1, j+1].axis('off')

    for i in range(transformed.shape[0]+1) :
        for j in range(transformed.shape[0]+1) :
            axes[i,j].axis('off')
    #plt.savefig("fig/" +str(c_idx) + "_result.jpg");
    plt.close('all')

    return rgba2rgb(figure_to_array(fig))

'''
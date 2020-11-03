import cv2
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy import interpolate

def get_differential_filter():
    filter_x=np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    filter_y=np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    return filter_x, filter_y

def filter_image(im, filter):
    im_filtered = np.zeros((np.size(im,0),np.size(im,1)))
    im_pad = np.pad(im,((1,1),(1,1)), 'constant')

    tracker_i=-1
    for i in range(np.size(im_filtered,0)):
        tracker_i+=1
        tracker_j=-1
        for j in range(np.size(im_filtered,1)):
            tracker_j+=1
            v=0
            for k in range(np.size(filter,0)):
                for l in range(np.size(filter,1)):
                    v += filter[k][l] * im_pad[k+tracker_i][l+tracker_j]
            im_filtered[i,j] = v
    return im_filtered

def find_match(img1, img2):
    sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=.02)
    kp1,des1 = sift.detectAndCompute(img1,None)
    kp2,des2 = sift.detectAndCompute(img2,None)
    neigh1 = NearestNeighbors(n_neighbors=2)
    neigh1.fit(des2)
    match1 = neigh1.kneighbors(des1) #[0]-distance [1]-index
    x1=[]
    x2=[]
    for i in range(np.size(match1[0],0)):
        if match1[0][i][0] < 0.7*match1[0][i][1]:
            x2.append(kp2[match1[1][i][0]].pt)
            x1.append(kp1[i].pt)
    x1 = np.floor(np.matrix(x1))
    x2 = np.floor(np.matrix(x2))

    return x1, x2

def align_image_using_feature(x1, x2, ransac_thr, ransac_iter):
    inliners = np.array([])
    As = np.zeros(((3,3,ransac_iter+1)))
    pad = np.ones(np.size(x1,0))
    pad = np.reshape(pad,(np.size(x1,0),1))
    x1_pad = np.append(x1,pad,axis=1)
    x2_pred = np.zeros((np.size(x2,0),3))
    iter = 0
    while (iter <= ransac_iter):
        r_i = np.random.choice(x1.shape[0],4,replace=False)
        r1 = x1[r_i, :]
        r2 = x2[r_i, :]

        temp = np.array(([r1[0,0],r1[0,1],1,0,0,0],[0,0,0,r1[0,0],r1[0,1],1]))
        for i in range(3):
            temp1 = np.array(([r1[i+1,0],r1[i+1,1],1,0,0,0],[0,0,0,r1[i+1,0],r1[i+1,1],1]))
            temp = np.append(temp,temp1)
        temp = np.reshape(temp,(8,6))
        r2 = np.reshape(r2,(8,1))
        A = np.matmul(np.matmul(np.linalg.inv(np.matmul(temp.transpose(),temp)),temp.transpose()),r2)
        temp2 = np.array(([[0],[0],[1]]))
        A = np.append(A,temp2,axis=0)
        A = np.reshape(A,(3,3))

        j=0
        for x in x1_pad:
            x2_p = np.matmul(A,x.transpose())
            x2_pred[j,:] = x2_p.transpose()
            j+=1
        x2_pad = np.append(x2,pad,axis=1)
        diff_x2 = abs(x2_pad-x2_pred)
        distances = np.linalg.norm(diff_x2, axis = 1)

        inliner = 0
        for distance in distances:
            if distance<=ransac_thr:
                inliner += 1
        # print("# of inliners: {}".format(inliner))
        inliners = np.append(inliners,inliner)
        As[:,:,iter] = A
        iter += 1

    max_iter = np.argmax(inliners)
    # print(max_iter)
    A = As[:,:,max_iter]
    return A

def warp_image(img, A, output_size):
    A_inv = np.linalg.inv(A)
    size_img_h = np.size(img,0)
    size_img_w = np.size(img,1)
    size_img = size_img_h * size_img_w
    H = np.arange(size_img_h)
    W = np.arange(size_img_w)

    size_t_h = output_size[0]
    size_t_w = output_size[1]
    size_t = size_t_h * size_t_w
    t_h = np.arange(size_t_h)
    t_w = np.arange(size_t_w)
    t_hh, t_ww = np.meshgrid(t_w,t_h)
    t_hh = np.reshape(t_hh,(size_t,1))
    t_ww = np.reshape(t_ww,(size_t,1))
    mesh_t = np.ones((size_t,3))
    mesh_t[:,0] = t_hh.transpose()
    mesh_t[:,1] = t_ww.transpose()

    inv_map = np.matmul(A,mesh_t.transpose())
    inv_map = inv_map.transpose()

    inv_map_f = np.ones((size_t,3))
    inv_map_f[:,1] = inv_map[:,0]
    inv_map_f[:,0] = inv_map[:,1]

    img_warped = interpolate.interpn((H,W), img, inv_map_f[:,0:2])
    img_warped = img_warped.reshape(size_t_h,size_t_w)

    return img_warped

def align_image(template, target, A):
    p1 = A[0,0] - 1
    p2 = A[0,1]
    p3 = A[0,2]
    p4 = A[1,0]
    p5 = A[1,1] - 1
    p6 = A[1,2]
    filter_x, filter_y = get_differential_filter()
    template_dx = filter_image(template,filter_x)
    template_dy = filter_image(template,filter_y)
    template_grad = np.array(([template_dx,template_dy]))

    u = np.arange(np.size(template,1))
    v = np.arange(np.size(template,0))
    u = np.tile(u,(np.size(template,0),1))
    v = np.tile(v,(np.size(template,1),1))
    v = v.transpose()
    one1 = np.ones((np.size(template,0), np.size(template,1)))
    zero1 = np.zeros((np.size(template,0), np.size(template,1)))

    dw_dp = np.array(([u, v, one1, zero1, zero1, zero1],[zero1, zero1, zero1, u, v, one1]))
    temp1 = np.multiply(template_dx, u)
    temp2 = np.multiply(template_dx, v)

    temp4 = np.multiply(template_dy, u)
    temp5 = np.multiply(template_dy, v)

    h1 = np.array(([temp1, temp2, template_dx, temp4, temp5, template_dy]))
    # Show gradI dw/dp #
    # f = plt.figure()
    # f.add_subplot(2,3,1)
    # plt.imshow(h1[0],vmin=np.min(h1[0]), vmax=np.max(h1[0]))
    # f.add_subplot(2,3,2)
    # plt.imshow(h1[1], vmin=np.min(h1[1]), vmax=np.max(h1[1]))
    # f.add_subplot(2,3,3)
    # plt.imshow(h1[2], vmin=np.min(h1[2]), vmax=np.max(h1[2]))
    # f.add_subplot(2,3,4)
    # plt.imshow(h1[3], vmin=np.min(h1[3]), vmax=np.max(h1[3]))
    # f.add_subplot(2,3,5)
    # plt.imshow(h1[4], vmin=np.min(h1[4]), vmax=np.max(h1[4]))
    # f.add_subplot(2,3,6)
    # plt.imshow(h1[5], vmin=np.min(h1[5]), vmax=np.max(h1[5]))
    # plt.show()

    H = np.zeros((6,6))
    count_i = 0
    for i in h1:
        count_j = 0
        for j in h1:
            H[count_i,count_j] = np.tensordot(i,j)
            count_j = count_j + 1
        count_i = count_i + 1

    P = np.array([p1,p2,p3,p4,p5,p6])
    P_mag = np.linalg.norm(P)
    F = np.zeros((6,1))
    iter=1

    H_inv = np.linalg.inv(H)
    dA = np.identity(3)
    dP_mag = 100000000000
    errors = 0
    while(dP_mag>0.001):
        img_warped = warp_image(target,A,template.shape)
        I_error = img_warped - template
        error = sum(sum(abs(I_error)))/(np.size(I_error,0)*np.size(I_error,1))
        count_i = 0
        for i in h1:
            F[count_i] = np.tensordot(i,I_error)
            count_i = count_i + 1

        delta_P = np.matmul(H_inv,F)
        dA[0,0] = 1 + delta_P[0]
        dA[0,1] = delta_P[1]
        dA[0,2] = delta_P[2]
        dA[1,0] = delta_P[3]
        dA[1,1] = 1 + delta_P[4]
        dA[1,2] = delta_P[5]

        dP_mag = np.linalg.norm(delta_P)
        iter+=1
        A = np.matmul(A,np.linalg.inv(dA))
        errors = np.append(errors,error)
    A_refined = A
    errors = np.delete(errors,0)
    return A_refined, errors


def track_multi_frames(template, img_list):
    ransac_thr = 5
    ransac_iter = 1000
    x1, x2 = find_match(template, img_list[0])
    A = align_image_using_feature(x1,x2,ransac_thr, ransac_iter)
    A_list = np.zeros((4,3,3))
    for i in range(4):
        print(i)
        A, errors = align_image(template, img_list[i], A)
        template = warp_image(img_list[i], A, template.shape)
        A_list[i] = A
    return A_list


def visualize_find_match(img1, img2, x1, x2, img_h=500):
    assert x1.shape == x2.shape, 'x1 and x2 should have same shape!'
    scale_factor1 = img_h/img1.shape[0]
    scale_factor2 = img_h/img2.shape[0]
    img1_resized = cv2.resize(img1, None, fx=scale_factor1, fy=scale_factor1)
    img2_resized = cv2.resize(img2, None, fx=scale_factor2, fy=scale_factor2)
    x1 = x1 * scale_factor1
    x2 = x2 * scale_factor2
    x2[:, 0] += img1_resized.shape[1]
    img = np.hstack((img1_resized, img2_resized))
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    for i in range(x1.shape[0]):
        plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'b')
        plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'bo')
    plt.axis('off')
    plt.show()

def visualize_align_image(template, target, A, A_refined, errors=None):
    img_warped_init = warp_image(target, A, template.shape)
    img_warped_optim = warp_image(target, A_refined, template.shape)
    err_img_init = np.abs(img_warped_init - template)
    err_img_optim = np.abs(img_warped_optim - template)
    img_warped_init = np.uint8(img_warped_init)
    img_warped_optim = np.uint8(img_warped_optim)
    overlay_init = cv2.addWeighted(template, 0.5, img_warped_init, 0.5, 0)
    overlay_optim = cv2.addWeighted(template, 0.5, img_warped_optim, 0.5, 0)
    plt.subplot(241)
    plt.imshow(template, cmap='gray')
    plt.title('Template')
    plt.axis('off')
    plt.subplot(242)
    plt.imshow(img_warped_init, cmap='gray')
    plt.title('Initial warp')
    plt.axis('off')
    plt.subplot(243)
    plt.imshow(overlay_init, cmap='gray')
    plt.title('Overlay')
    plt.axis('off')
    plt.subplot(244)
    plt.imshow(err_img_init, cmap='jet')
    plt.title('Error map')
    plt.axis('off')
    plt.subplot(245)
    plt.imshow(template, cmap='gray')
    plt.title('Template')
    plt.axis('off')
    plt.subplot(246)
    plt.imshow(img_warped_optim, cmap='gray')
    plt.title('Opt. warp')
    plt.axis('off')
    plt.subplot(247)
    plt.imshow(overlay_optim, cmap='gray')
    plt.title('Overlay')
    plt.axis('off')
    plt.subplot(248)
    plt.imshow(err_img_optim, cmap='jet')
    plt.title('Error map')
    plt.axis('off')
    plt.show()

    if errors is not None:
        plt.plot(errors * 255)
        plt.xlabel('Iteration')
        plt.ylabel('Error')
        plt.show()


def visualize_track_multi_frames(template, img_list, A_list):
    bbox_list = []
    for A in A_list:
        boundary_t = np.hstack((np.array([[0, 0], [template.shape[1], 0], [template.shape[1], template.shape[0]],
                                        [0, template.shape[0]], [0, 0]]), np.ones((5, 1)))) @ A[:2, :].T
        bbox_list.append(boundary_t)

    plt.subplot(221)
    plt.imshow(img_list[0], cmap='gray')
    plt.plot(bbox_list[0][:, 0], bbox_list[0][:, 1], 'r')
    plt.title('Frame 1')
    plt.axis('off')
    plt.subplot(222)
    plt.imshow(img_list[1], cmap='gray')
    plt.plot(bbox_list[1][:, 0], bbox_list[1][:, 1], 'r')
    plt.title('Frame 2')
    plt.axis('off')
    plt.subplot(223)
    plt.imshow(img_list[2], cmap='gray')
    plt.plot(bbox_list[2][:, 0], bbox_list[2][:, 1], 'r')
    plt.title('Frame 3')
    plt.axis('off')
    plt.subplot(224)
    plt.imshow(img_list[3], cmap='gray')
    plt.plot(bbox_list[3][:, 0], bbox_list[3][:, 1], 'r')
    plt.title('Frame 4')
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    template_path = ''
    template = cv2.imread(template_path, 0)  # read as grey scale image
    target_list = []

    for i in range(4):
        target = cv2.imread('./target{}.jpg'.format(i+1), 0)  # read as grey scale image
        target_list.append(target)

    # x1, x2 = find_match(template, target_list[0])
    # visualize_find_match(template, target_list[0], x1, x2)
    # ransac_thr = 5
    # ransac_iter = 100
    # A = align_image_using_feature(x1, x2, ransac_thr, ransac_iter)

    # img_warped = warp_image(target_list[0], A, template.shape)
    # plt.imshow(img_warped, cmap='gray', vmin=0, vmax=255)
    # plt.axis('off')
    # plt.show()

    # A_refined, errors = align_image(template, target_list[0], A)
    # visualize_align_image(template, target_list[0], A, A_refined, errors)
    A_list = track_multi_frames(template, target_list)
    visualize_track_multi_frames(template, target_list, A_list)

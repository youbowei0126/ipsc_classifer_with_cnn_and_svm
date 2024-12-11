import matplotlib.pyplot as plt
import joblib
import numpy as np
from tqdm import tqdm
x_test_embedded=joblib.load(r"data_in_model007\x_test_embedded.joblib")
x_test_pca=joblib.load(r"data_in_model007\x_test_pca.joblib")
x_test_lda=joblib.load(r"data_in_model007\x_test_lda.joblib")
x_train_embedded=joblib.load(r"data_in_model007\x_train_embedded.joblib")
x_train_pca=joblib.load(r"data_in_model007\x_train_pca.joblib")
x_train_lda=joblib.load(r"data_in_model007\x_train_lda.joblib")
y_test_cata=joblib.load(r"data_in_model007\y_test_cata.joblib")
y_train_cata=joblib.load(r"data_in_model007\y_train_cata.joblib")


from umap import UMAP
umap = UMAP(n_components=3, random_state=42)

'''fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_train_embedded[:,0],x_train_embedded[:,1],x_train_embedded[:,2],c=y_train_cata,cmap='coolwarm',s=5)
ax.scatter(x_test_embedded[:,0],x_test_embedded[:,1],x_test_embedded[:,2],c=y_test_cata,cmap='coolwarm',s=30)


fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_train_pca[:,0],x_train_pca[:,1],x_train_pca[:,2],c=y_train_cata,cmap='coolwarm',s=5)
ax.scatter(x_test_pca[:,0],x_test_pca[:,1],x_test_pca[:,2],c=y_test_cata,cmap='coolwarm',s=30)'''


def render_anime(x_train_lda,x_test_lda,y_train_cata,y_test_cata,path_):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_train_lda[:,0],x_train_lda[:,1],x_train_lda[:,2],c=y_train_cata,cmap='coolwarm',s=5)
    ax.scatter(x_test_lda[:,0],x_test_lda[:,1],x_test_lda[:,2],c=y_test_cata,cmap='coolwarm',s=30)


    import matplotlib.animation as animation

    # 定義更新函數，讓視角旋轉
    def update(frame):
        ax.view_init(elev=30, azim=frame)  # 調整高度和方位角
        return fig,

    # 創建動畫
    ani = animation.FuncAnimation(fig, update, frames=tqdm(range(0, 360, 1)), interval=25)

    # 保存為 MP4 或 GIF
    ani.save(path_+".mp4", writer="ffmpeg", fps=30,dpi=600)  # 保存為 MP4
    ani.save(path_+".gif", writer="imagemagick", fps=30,dpi=300)  # 保存為 GIF




render_anime(x_train_embedded,x_test_embedded,y_train_cata,y_test_cata,r"data_in_model007\umap")
render_anime(x_train_pca,x_test_pca,y_train_cata,y_test_cata,r"data_in_model007\pca")
render_anime(x_train_lda,x_test_lda,y_train_cata,y_test_cata,r"data_in_model007\lda")
plt.show()
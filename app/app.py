from flask import Flask, render_template, Response, request, send_file, session,jsonify

from scipy.linalg import eigh
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.datasets import make_moons
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.gaussian_process.kernels import RBF
import io
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

app = Flask(__name__)
length_scale = 0.1
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'  # Be sure to set a secret key
base_plot_sample_size = 20
mask_scale = 250


def set_mask_scale(sample_size):
    if sample_size > 0 and sample_size <=24:
        return 250
    elif sample_size > 24 and sample_size <= 32:
        return 5
    else:
        return 1
    
# sample_size = session.get('sample_size')
# num_clusters = session.get('num_clusters')


@app.route('/plot_step1/<int:sample_size>/<int:num_clusters>/<float:noise>')
def plot_step1(sample_size, num_clusters, noise):

    X, y = make_moons(n_samples=sample_size, noise=noise, random_state=0)
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    xs, ys = zip(*X)
    axis.scatter(xs, ys)
    axis.set_title('Sample Data')
    for i, (x, y) in enumerate(X):
        axis.text(x, y, str(i), fontsize=12)

    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')


@app.route('/step1', methods=['POST'])
def image():
    num_clusters = int(request.form['clusters'])
    sample_size = int(request.form['sample_size'])
    noise = float(request.form['noise'])
    session['sample_size'] = sample_size
    session['num_clusters'] = num_clusters
    session['noise'] = noise
    return render_template('step1.html', num_clusters=num_clusters, sample_size=sample_size, noise=noise)


@app.route('/plot_step2')
def plot_step2():
    num_clusters = session.get('num_clusters')
    sample_size = session.get('sample_size')
    mask_scale = set_mask_scale(sample_size)
    noise = session.get('noise')
    X, _ = make_moons(n_samples=sample_size, noise=noise, random_state=0)
    kernel = RBF(length_scale=length_scale)
    affinity_matrix = kernel(X)
    # Scaling non-diagonal elements of the affinity matrix by a factor of 100
    i = np.arange(affinity_matrix.shape[0])
    j = np.arange(affinity_matrix.shape[1])
    mask = (i[:, None] != j)
    affinity_matrix[mask] *= mask_scale

    A = pd.DataFrame(affinity_matrix)

    matrix_plot_size_scale = (sample_size) / base_plot_sample_size
    fig = Figure(figsize=(12*matrix_plot_size_scale,
                 9*matrix_plot_size_scale))
    axs = fig.add_subplot(1, 1, 1)

    # Set ticks to have interval of 1
    axs.set_xticks(np.arange(0, A.shape[0], 1))
    axs.set_yticks(np.arange(0, A.shape[1], 1))

    # Replace seaborn heatmap with matplotlib matshow
    cax = axs.imshow(A, cmap='coolwarm')
    fig.colorbar(cax)
    axs.set_title("Affinity Matrix")

    # iterate over all cells and add text annotation
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            axs.text(j, i, format(A.values[i, j], ".2f"),
                     ha="center", va="center", color="black")
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')


@app.route('/step2')
def step2():
    return render_template('step2.html')


@app.route('/plot_step3_1')
def plot_step3_1():
    num_clusters = session.get('num_clusters')
    sample_size = session.get('sample_size')
    noise = session.get('noise')
    mask_scale = set_mask_scale(sample_size)

    X, _ = make_moons(n_samples=sample_size, noise=noise, random_state=0)
    kernel = RBF(length_scale=length_scale)
    affinity_matrix = kernel(X)
    # Scaling non-diagonal elements of the affinity matrix by a factor of x
    i = np.arange(affinity_matrix.shape[0])
    j = np.arange(affinity_matrix.shape[1])
    mask = (i[:, None] != j)
    affinity_matrix[mask] *= mask_scale

    D = np.diag(np.round(np.sum(affinity_matrix, axis=1), 2))

    # Increase DPI for a more detailed image
    matrix_plot_size_scale = (sample_size) / base_plot_sample_size
    fig = Figure(figsize=(12*matrix_plot_size_scale,
                 9*matrix_plot_size_scale), dpi=200)
    axs = fig.add_subplot(1, 1, 1)
    # Set ticks to have interval of 1
    axs.set_xticks(np.arange(0, D.shape[0], 1))
    axs.set_yticks(np.arange(0, D.shape[1], 1))

    cax = axs.imshow(D, cmap='coolwarm')
    fig.colorbar(cax)
    axs.set_title("Diagonal Matrix")

    for (i, j), z in np.ndenumerate(D):
        axs.text(j, i, str(round(z, 2)), ha='center', va='center')

    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')


@app.route('/plot_step3_2')
def plot_step3_2():
    num_clusters = session.get('num_clusters')
    sample_size = session.get('sample_size')
    noise = session.get('noise')
    mask_scale = set_mask_scale(sample_size)

    X, _ = make_moons(n_samples=sample_size, noise=noise, random_state=0)
    kernel = RBF(length_scale=length_scale)
    affinity_matrix = kernel(X)
    # Scaling non-diagonal elements of the affinity matrix by a factor of x
    i = np.arange(affinity_matrix.shape[0])
    j = np.arange(affinity_matrix.shape[1])
    mask = (i[:, None] != j)
    affinity_matrix[mask] *= mask_scale

    D = np.diag(np.round(np.sum(affinity_matrix, axis=1), 2))
    D_inverse = np.linalg.inv(D)
    D_powered = np.sqrt(D_inverse)

    L = np.matmul(D_powered, affinity_matrix)
    L = np.matmul(L, D_powered)
    # Increase DPI for a more detailed image
    matrix_plot_size_scale = (sample_size) / base_plot_sample_size
    fig = Figure(figsize=(12*matrix_plot_size_scale,
                 9*matrix_plot_size_scale), dpi=200)
    axs = fig.add_subplot(1, 1, 1)

    # Set ticks to have interval of 1
    axs.set_xticks(np.arange(0, D.shape[0], 1))
    axs.set_yticks(np.arange(0, D.shape[1], 1))

    cax = axs.imshow(L, cmap='coolwarm')

    fig.colorbar(cax)
    axs.set_title("Laplacian Matrix")


    for (i, j), z in np.ndenumerate(L):
        axs.text(j, i, str(round(z, 2)), ha='center', va='center')

    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')


@app.route('/step3')
def step3():
    return render_template('step3.html')


@app.route('/step4')
def step4():
    num_clusters = session.get('num_clusters')
    sample_size = session.get('sample_size')
    noise = session.get('noise')
    X, _ = make_moons(n_samples=sample_size, noise=noise, random_state=0)


    kernel = RBF(length_scale=length_scale)
    affinity_matrix = kernel(X)

    D = np.diag(np.round(np.sum(affinity_matrix, axis=1), 2))
    D_inverse = np.linalg.inv(D)
    D_powered = np.sqrt(D_inverse)
    L = np.matmul(D_powered, affinity_matrix)
    L = np.matmul(L, D_powered)

    eigenvalues, eigenvectors = eigh(L)
    print("eigenvalues")
    print(eigenvalues)
    print("eigenvectors")
    print(eigenvectors)
    # get the indices that would sort the eigenvalues from scipy.linalg.eigh in the same order as those from numpy.linalg.eig
    sort_indices = np.argsort(eigenvalues)
    # descending order
    sort_indices = sort_indices[::-1]
    # use these indices to sort the eigenvectors from scipy.linalg.eigh
    sorted_eigenvalues = eigenvalues[sort_indices]
    sorted_eigenvalues = sorted_eigenvalues[:num_clusters]
    sorted_eigenvectors = eigenvectors[:, sort_indices]
    # sorted_eigenvectors = sorted_eigenvectors[:num_clusters]
    
    # select the first 'num_clusters' columns from the sorted eigenvectors
    Y = sorted_eigenvectors[:, :num_clusters]

    Y_normalized = normalize(Y, axis=1, norm='l2')
    # print("Y")
    # print(Y)
    # print("Y_normalized")
    # print(Y_normalized)
    # print("sorted_eigenvalues")
    # print(sorted_eigenvectors)
    return render_template('step4.html', eigenvalues=sorted_eigenvalues, eigenvectors=sorted_eigenvectors[:, :num_clusters],
                           Y_normalized=Y_normalized)


@app.route('/step5')
def step5():
    return render_template('step5.html')


@app.route('/plot_step5')
def plot_step5():

    num_clusters = session.get('num_clusters')

    sample_size = session.get('sample_size')

    noise = session.get('noise')


    X, _ = make_moons(n_samples=sample_size, noise=noise, random_state=0)


    kernel = RBF(length_scale=length_scale)
    affinity_matrix = kernel(X)

    D = np.diag(np.round(np.sum(affinity_matrix, axis=1), 2))
    D_inverse = np.linalg.inv(D)
    D_powered = np.sqrt(D_inverse)
    L = np.matmul(D_powered, affinity_matrix)
    L = np.matmul(L, D_powered)

    eigenvalues, eigenvectors = eigh(L)

    Y = eigenvectors[:, :num_clusters]
    # get the indices that would sort the eigenvalues from scipy.linalg.eigh in the same order as those from numpy.linalg.eig
    sort_indices = np.argsort(eigenvalues)
    # descending order
    sort_indices = sort_indices[::-1]
    # use these indices to sort the eigenvectors from scipy.linalg.eigh
    sorted_eigenvectors = eigenvectors[:, sort_indices]
    
    # select the first 'num_clusters' columns from the sorted eigenvectors
    Y = sorted_eigenvectors[:, :num_clusters]

    Y_normalized = normalize(Y, axis=1, norm='l2')

    kmeans = KMeans(n_clusters=num_clusters, max_iter=1000)
    kmeans.fit(Y_normalized)

    labels = kmeans.labels_


    fig = Figure(figsize=(10, 10))
    axs = fig.add_subplot(1, 1, 1)
    axs.scatter(X[:, 0], X[:, 1], c=labels)
    axs.set_title('Clustered Graph')
    for i, coord in enumerate(X):
        coord_int = coord.astype(int)
        axs.annotate(f'{i}',
                     (coord[0], coord[1]),
                     textcoords="offset points",
                     xytext=(-10, -10),
                     ha='center')

    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')


@app.route('/')
def home():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)

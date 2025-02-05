import numpy as np
import pandas as pd
import io
from flask import Flask, render_template, Response, request, session, redirect, url_for
from scipy.linalg import eigh
from sklearn.datasets import make_moons
from sklearn.datasets import make_circles
from sklearn.datasets import make_blobs
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.gaussian_process.kernels import RBF
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

app = Flask(__name__)
# 0.11 is good until 23
# 0.12 is good until 17
# 0.14 is good until 15
# length_scale = 0.14
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/' 
base_plot_sample_size = 20
mask_scale = 250

def make_data(data_set,sample_size, noise, num_clusters):

    if data_set == 'make_circles':
        X, _ = make_circles(n_samples=sample_size, factor=0.3, noise=noise)
        return X, _
    elif data_set == 'make_moons':
        X, _ = make_moons(n_samples=sample_size, noise=noise, random_state=0)
        return X, _
    elif data_set == 'make_blobs':
        X, _ = make_blobs(n_samples=sample_size, centers=num_clusters, cluster_std=0.2, random_state=0)
        return X, _

def set_mask_scale(sample_size):
    if sample_size > 0 and sample_size <=24:
        return 10
    elif sample_size > 24 and sample_size <= 32:
        return 2.5
    else:
        return 1


@app.route('/index', methods=['POST'])
def handle_form():
    num_clusters = int(request.form['clusters'])
    sample_size = int(request.form['sample_size'])
    display_mode = request.form.get('display_mode')  # 'on' if checked, None if not checked
    noise = float(request.form['noise'])
    dataset_choice = request.form.get('dataset_choice', 'make_moons')
    length_scale_value = request.form.get('length_scale', '0.14')  # Attempt to get value or default to '0.14'
    if not length_scale_value:  # Check if the value is empty
        length_scale_value = '0.14'
    length_scale = float(length_scale_value)

    session['dataset_choice'] = dataset_choice
    session['sample_size'] = sample_size
    session['num_clusters'] = num_clusters
    session['noise'] = noise
    session['display_mode'] = display_mode
    session['length_scale'] = length_scale
    if display_mode == 'on':
        return redirect(url_for('step1', sample_size=sample_size, num_clusters=num_clusters, noise=noise))
    else:
        return redirect(url_for('all_steps'))

    

@app.route('/plot_step1/<int:sample_size>/<int:num_clusters>/<float:noise>/<string:dataset_choice>')
def plot_step1(sample_size, num_clusters, noise, dataset_choice):

    X, y = make_data(dataset_choice,sample_size, noise, num_clusters)

    fig = Figure(facecolor='grey')
    axs = fig.add_subplot(1, 1, 1, facecolor='#d3d3d3')

    axs.scatter(X[:, 0], X[:, 1])
    axs.set_title('Data Graph')

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


@app.route('/step1', methods=['POST', 'GET'])
def step1():
    num_clusters = session.get('num_clusters')
    sample_size = session.get('sample_size')
    noise = session.get('noise')
    dataset_choice = session.get('dataset_choice')
    return render_template('step1.html', num_clusters=num_clusters, sample_size=sample_size, noise=noise, dataset_choice=dataset_choice)



@app.route('/plot_step2')
def plot_step2():
    num_clusters = session.get('num_clusters')
    sample_size = session.get('sample_size')
    dataset_choice = session.get('dataset_choice')
    noise = session.get('noise')
    mask_scale = set_mask_scale(sample_size)
    length_scale = session.get('length_scale')
    X, y = make_data(dataset_choice,sample_size, noise, num_clusters)
    # X, _ = make_moons(n_samples=sample_size, noise=noise, random_state=0)
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
                 9*matrix_plot_size_scale), facecolor='grey')
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
    dataset_choice = session.get('dataset_choice')
    mask_scale = set_mask_scale(sample_size)
    length_scale = session.get('length_scale')
    X, y = make_data(dataset_choice,sample_size, noise, num_clusters)
    # X, _ = make_moons(n_samples=sample_size, noise=noise, random_state=0)
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
                 9*matrix_plot_size_scale), dpi=200, facecolor='grey')
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
    dataset_choice = session.get('dataset_choice')
    mask_scale = set_mask_scale(sample_size)
    length_scale = session.get('length_scale')
    X, y = make_data(dataset_choice,sample_size, noise, num_clusters)
    # X, _ = make_moons(n_samples=sample_size, noise=noise, random_state=0)
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
                 9*matrix_plot_size_scale), dpi=200, facecolor='grey')
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
    dataset_choice = session.get('dataset_choice')
    length_scale = session.get('length_scale')
    X, y = make_data(dataset_choice,sample_size, noise, num_clusters)

    kernel = RBF(length_scale=length_scale)
    affinity_matrix = kernel(X)

    D = np.diag(np.round(np.sum(affinity_matrix, axis=1), 2))
    D_inverse = np.linalg.inv(D)
    D_powered = np.sqrt(D_inverse)
    L = np.matmul(D_powered, affinity_matrix)
    L = np.matmul(L, D_powered)

    eigenvalues, eigenvectors = eigh(L)

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
    # Convert the NumPy arrays to lists of lists for Jinja2
    sorted_eigenvalues = sorted_eigenvalues.tolist()
    sorted_eigenvectors = [list(vec) for vec in sorted_eigenvectors[:, :num_clusters].T]
    Y_normalized = Y_normalized.T # Transpose Y_normalized before converting to list of lists
    Y_normalized = [list(vec) for vec in Y_normalized]

    return render_template('step4.html', eigenvalues=sorted_eigenvalues, eigenvectors=sorted_eigenvectors,
                           Y_normalized=Y_normalized)
@app.route('/plot_step5')
def plot_step5():

    num_clusters = session.get('num_clusters')

    sample_size = session.get('sample_size')

    noise = session.get('noise')
    dataset_choice = session.get('dataset_choice')
    length_scale = session.get('length_scale')
    if(num_clusters >2):
        return None
    
    X, y = make_data(dataset_choice,sample_size, noise, num_clusters)
    # X, _ = make_moons(n_samples=sample_size, noise=noise, random_state=0)


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

    cluster_centers_ = kmeans.cluster_centers_
    print(cluster_centers_)
    fig = Figure(figsize=(10, 10), facecolor='grey')
    axs = fig.add_subplot(1, 1, 1,facecolor='#d3d3d3')
    axs.scatter(Y_normalized[:, 0], Y_normalized[:, 1])
    axs.scatter(cluster_centers_[:, 0], cluster_centers_[:, 1], c='red')
    axs.set_title('Eigenvectors-point Plot')
    for i, coord in enumerate(Y_normalized):
        coord_int = coord.astype(int)
        axs.annotate(f'{i}',
                     (coord[0], coord[1]),
                     textcoords="offset points",
                     xytext=(-10, -10),
                     ha='center')
    for i, coord in enumerate(cluster_centers_):
        coord_int = coord.astype(int)
        axs.annotate('c'+f'{i}',
                     (coord[0], coord[1]),
                     textcoords="offset points",
                     xytext=(-10, -10),
                     ha='center')
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

@app.route('/step5')
def step5():
    num_clusters = session.get('num_clusters')
    sample_size = session.get('sample_size')
    noise = session.get('noise')
    dataset_choice = session.get('dataset_choice')

    return render_template('step5.html', sample_size=sample_size, noise=noise, num_clusters=num_clusters, dataset_choice=dataset_choice)


@app.route('/step6')
def step6():
    return render_template('step6.html')


@app.route('/plot_step6')
def plot_step6():

    num_clusters = session.get('num_clusters')

    sample_size = session.get('sample_size')

    noise = session.get('noise')
    dataset_choice = session.get('dataset_choice')
    length_scale = session.get('length_scale')
    X, y = make_data(dataset_choice,sample_size, noise, num_clusters)
    # X, _ = make_moons(n_samples=sample_size, noise=noise, random_state=0)


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


    fig = Figure(figsize=(10, 10), facecolor='grey')
    axs = fig.add_subplot(1, 1, 1,facecolor='#d3d3d3')
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

@app.route('/all_steps')
def all_steps():
    num_clusters = session.get('num_clusters')
    sample_size = session.get('sample_size')
    noise = session.get('noise')
    dataset_choice = session.get('dataset_choice')
    length_scale = session.get('length_scale')
    
    # Plot the dataset
    X, _ = make_data(dataset_choice,sample_size, noise, num_clusters)

    # X, _ = make_moons(n_samples=sample_size, noise=noise, random_state=0)

    kernel = RBF(length_scale=length_scale)
    affinity_matrix = kernel(X)

    D = np.diag(np.round(np.sum(affinity_matrix, axis=1), 2))
    D_inverse = np.linalg.inv(D)
    D_powered = np.sqrt(D_inverse)
    L = np.matmul(D_powered, affinity_matrix)
    L = np.matmul(L, D_powered)

    eigenvalues, eigenvectors = eigh(L)

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
    kmeans = KMeans(n_clusters=num_clusters, max_iter=1000)
    kmeans.fit(Y_normalized)

    labels = kmeans.labels_
    # Convert the NumPy arrays to lists of lists for Jinja2
    sorted_eigenvalues = sorted_eigenvalues.tolist()
    sorted_eigenvectors = [list(vec) for vec in sorted_eigenvectors[:, :num_clusters].T]
    Y_normalized = Y_normalized.T # Transpose Y_normalized before converting to list of lists
    Y_normalized = [list(vec) for vec in Y_normalized]

    return render_template('all.html', eigenvalues=sorted_eigenvalues, eigenvectors=sorted_eigenvectors,
                           Y_normalized=Y_normalized, num_clusters=num_clusters, sample_size=sample_size, noise=noise, labels=labels, dataset_choice=dataset_choice)



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/credit')
def credit():
  return render_template('credits.html')



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
    # app.run(debug=True, use_reloader=True)

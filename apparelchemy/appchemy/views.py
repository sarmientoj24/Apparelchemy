from django.shortcuts import render
from django.http import JsonResponse
from django.http import HttpResponse
import random
import numpy as np
from PIL import Image
import sqlite3, glob, os,  operator, math
import pandas as pd
from keras.losses import mse, binary_crossentropy
from keras import backend as K
from keras.models import load_model
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
import base64
import json as simplejson

# Import model here

############ VARIABLE DECLARATIONS
cwd = os.getcwd()
z_dim = 16
path_to_static = '/appchemy/static/appchemy'
db_file_path = cwd + path_to_static + '/' + "db.sqlite3"
use_fixed_epsilon=True        # WE WONT USE FIXED EPSILON FIRST

##################### MODEL LOADING AND VARIABLE LOADING ON HOMEPAGE LOAD, LOADS ONCE
decoder_loaded = load_model(cwd + path_to_static + '/decoder.h5')
decoder_loaded._make_predict_function()
graph = tf.get_default_graph()

cluster_centroids = None
product_df = None
use_fixed_eps_cl = False

try:
    conn = sqlite3.connect(db_file_path)
    if use_fixed_eps_cl:
        sql = '''
            SELECT cluster_id, 'z_fixed_code_1', 'z_fixed_code_2', 'z_fixed_code_3', 'z_fixed_code_4', 
                'z_fixed_code_5', 'z_fixed_code_6', 'z_fixed_code_7', 'z_fixed_code_8', 
                'z_fixed_code_9', 'z_fixed_code_10', 'z_fixed_code_11', 'z_fixed_code_12', 
                'z_fixed_code_13', 'z_fixed_code_14', 'z_fixed_code_15', 'z_fixed_code_16'
            FROM cluster_centroids
            '''
    else:
        sql = '''
        SELECT cluster_id, z_mean_1, z_mean_2, z_mean_3, z_mean_4, 
            z_mean_5, z_mean_6, z_mean_7, z_mean_8, 
            z_mean_9, z_mean_10, z_mean_11, z_mean_12, 
            z_mean_13, z_mean_14, z_mean_15, z_mean_16
        FROM cluster_centroids
        '''

    cluster_centroids = pd.read_sql_query(sql, conn)
    if use_fixed_epsilon:
        sql = '''
            SELECT img_id, cluster_id, z_fixed_code_1, z_fixed_code_2, z_fixed_code_3, z_fixed_code_4, 
                    z_fixed_code_5, z_fixed_code_6, z_fixed_code_7, z_fixed_code_8, 
                    z_fixed_code_9, z_fixed_code_10, z_fixed_code_11, z_fixed_code_12, 
                    z_fixed_code_13, z_fixed_code_14, z_fixed_code_15, z_fixed_code_16
            FROM product_entries
        '''
    else:
        sql = '''
            SELECT img_id, cluster_id, z_mean_1, z_mean_2, z_mean_3, z_mean_4, 
                z_mean_5, z_mean_6, z_mean_7, z_mean_8, 
                z_mean_9, z_mean_10, z_mean_11, z_mean_12, 
                z_mean_13, z_mean_14, z_mean_15, z_mean_16,
                z_log_var_1, z_log_var_2, z_log_var_3, z_log_var_4, 
                z_log_var_5, z_log_var_6, z_log_var_7, z_log_var_8, 
                z_log_var_9, z_log_var_10, z_log_var_11, z_log_var_12, 
                z_log_var_13, z_log_var_14, z_log_var_15, z_log_var_16
            FROM product_entries
        '''
    product_df = pd.read_sql_query(sql, conn)           # CONTAINS ALL PRODUCTS IN-MEMORY
    conn.close()
except Exception as e:
    print(str(e))

if use_fixed_eps_cl:
    centroids_df_columns_vec_diff = ['z_fixed_code_1', 'z_fixed_code_2', 'z_fixed_code_3', 'z_fixed_code_4', 
        'z_fixed_code_5', 'z_fixed_code_6', 'z_fixed_code_7', 'z_fixed_code_8',
        'z_fixed_code_9', 'z_fixed_code_10', 'z_fixed_code_11', 'z_fixed_code_12',
        'z_fixed_code_13', 'z_fixed_code_14', 'z_fixed_code_15', 'z_fixed_code_16']
else:
    centroids_df_columns_vec_diff = ['z_mean_1', 'z_mean_2', 'z_mean_3', 'z_mean_4',
        'z_mean_5', 'z_mean_6', 'z_mean_7', 'z_mean_8',
        'z_mean_9', 'z_mean_10', 'z_mean_11', 'z_mean_12',
        'z_mean_13', 'z_mean_14', 'z_mean_15', 'z_mean_16']

centroids = cluster_centroids[centroids_df_columns_vec_diff].to_numpy() # used fpr predicting clusters
centroids_difference = []       # Use this hopefully for recommendation, vector projection


random_zs = [[-2.73730707168579, -0.466104239225388, 1.10926961898804, 0.450847536325455, -0.252695083618164, -1.23918509483337, 0.117246024310589, 1.65934920310974, 1.19904482364655, -0.225397750735283, -0.708985686302185, -0.271019816398621, 2.28351044654846, -1.9463746547699, -0.400244206190109, -0.182484835386276],
        [-0.731075167655945, 0.776988804340363, -1.27827870845795, 0.395890980958939, 0.335724830627441, 2.29909920692444, -1.92729115486145, -0.171787783503532, 0.584947824478149, -1.69782590866089, -0.823324859142303, -1.20246553421021, 1.16204845905304, 0.763690412044525, 0.587404489517212, 0.994596779346466],
        [0.35709872841835, 2.45453333854675, 0.549497067928314, 0.381065636873245, 0.159956693649292, -0.411005407571793, 0.0273209735751152, -2.61430430412292, 1.81841421127319, -0.764032363891602, 0.0651724636554718, -3.65531349182129, -1.0060772895813, -0.0383533164858818, 0.725042283535004, -0.929942727088928],
        [0.224376186728477, 1.10158157348633, -0.0106806457042694, -1.26896893978119, -1.61814665794373, 0.554606139659882, 1.22298669815063, -0.778548717498779, 0.34887433052063, -0.0201060175895691, 1.54122447967529, -1.35477674007416, 0.277720093727112, -1.37040781974792, 0.972314655780792, -0.306051254272461],
        [-1.59616422653198, 0.288856536149979, -0.818448066711426, 2.49770903587341, 0.188682600855827, -0.749235808849335, 2.04487156867981, 0.0165311098098755, 1.56475615501404, 0.101957648992538, 0.881394505500793, -0.0600893795490265, -0.709332287311554, 0.267514079809189, -0.536377906799316, -0.0400786101818085],
        [-1.59616422653198, 0.288856536149979, -0.818448066711426, 2.49770903587341, 0.188682600855827, -0.749235808849335, 2.04487156867981, 0.0165311098098755, 1.56475615501404, 0.101957648992538, 0.881394505500793, -0.0600893795490265, -0.709332287311554, 0.267514079809189, -0.536377906799316, -0.0400786101818085],
        [-1.20985054969788, -2.94175219535828, -0.892850637435913, 0.463192820549011, 0.118247449398041, -1.53433656692505, -0.941357910633087, -0.0232774019241333, 1.23727011680603, 1.11509943008423, -1.27888572216034, 0.655416905879974, 0.135799840092659, 0.989586174488068, 0.220343515276909, 1.24712002277374],
        [-0.0514819398522377, 0.0268264040350914, -0.0875119119882584, 0.39195916056633, 0.161404147744179, -2.13854789733887, -1.41359758377075, 0.245206564664841, 2.33376407623291, -0.889240562915802, -0.991460680961609, 0.807750701904297, 1.2171049118042, 0.0760719627141953, 0.849676251411438, 0.787343800067902],
        [1.0904655456543, 0.450513005256653, 0.135962396860123, -0.0852451249957085, -1.71725952625275, 0.586381018161774, 1.10269284248352, 0.710701107978821, 0.453411221504211, -0.0756871253252029, -0.0982975289225578, -1.80453157424927, -1.33362257480621, 0.305797904729843, -0.815776586532593, 2.26736259460449],
        [0.667591750621796, 0.425458788871765, 0.101126685738564, -0.905273735523224, -0.586958289146423, 1.0243661403656, 0.952484011650085, 1.09352123737335, 0.597325921058655, -0.0250408947467804, 0.706064581871033, 1.66244602203369, -1.50268959999084, -0.944395422935486, 0.292690336704254, -0.191049352288246],
        [1.32861471176147, -0.0716913044452667, -0.62359619140625, -1.71819233894348, 0.0196839720010757, -0.963969826698303, -0.361377000808716, -0.191044867038727, 1.88396453857422, -0.387775719165802, 1.99360072612762, -0.497542858123779, 0.62942111492157, 1.35984253883362, -0.456715077161789, -1.61969470977783],
        [0.049395464360714, -0.857829928398132, -1.67249953746796, 3.23793387413025, 0.273747891187668, 0.375442951917648, 1.53254866600037, 0.604103207588196, 1.69045162200928, -1.30021166801453, 0.848221242427826, -0.051739901304245, 0.970500588417053, -2.35861134529114, 0.00210124254226685, -0.672622263431549],
        [0.501738846302032, 0.0395405068993568, -0.615134119987488, 2.51165866851807, 0.944608390331268, 0.726586401462555, 1.62455725669861, 0.414975702762604, 0.653059601783752, 1.30641317367554, 0.549112915992737, 0.11162331700325, 0.732404768466949, -3.3315167427063, 0.749285697937012, -1.44883489608765]
    ]
random_z = random_zs[random.randint(0, 12)]

####### END OF VARIABLE DECLARATIONS

####### FUNCTION DECLARATIONS
def get_vector_difference(arr1, arr2):
    diff = []
    for i in range(len(arr1)):
        diff.append(arr1[i] - arr2[i])
    return diff

centroids_row, centroids_cols = centroids.shape

for i in range(centroids_row):
    inside_dim = []
    for j in range(centroids_row):
        if i == j:
            inside_dim.append([0] * centroids_cols)
        else:
            inside_dim.append(get_vector_difference(centroids[i], centroids[j]))
    centroids_difference.append(inside_dim)

def compute_log_likelihood(z_gen, mu_, log_var):
    var_ = np.exp(log_var)
    sigma = np.sqrt(var_)
    term_a = (z_gen - mu_) ** 2
    term_b = 2 * (sigma ** 2)
    term_c = -1 * (term_a / term_b)
    # −log(2π−−√σi(x))
    term_d = ((2 * np.pi) ** 0.5) * sigma
    np.log(term_d)
    term_e = term_c - term_d
    likelihood = np.sum(term_e)
    return likelihood

def get_distance(metric='euclidean', z_=None, data_vector=None):
    if metric == 'euclidean':
        return np.linalg.norm(z_ - data_vector)# lowest is highest
    elif use_fixed_epsilon == False and metric == 'max_likelihood':
    # only USE this for use_fixed_epsilon = False
        return compute_log_likelihood(z_, data_vector[:16], data_vector[16:]) * -1    # parang mali need itest
    else:
        return np.linalg.norm(z_ - data_vector, ord=1)# lowest is highest

def get_nearest_image(z_, dataframe_points, get_top=False, metrics='max_likelihood'):
    img_ids = dataframe_points['img_id'].to_list()
    columns_needed = [i for i in dataframe_points.columns.values.tolist() if i != 'img_id']
    data = dataframe_points[columns_needed].to_numpy()

    # do for all
    rows = len(data)
    distances = {}
    for i in range(rows):
        distances[img_ids[i]] = get_distance(metric=metrics, z_=z_, data_vector=data[i])
    # Sort distances, return top 3 keys() Get highest ones
    distances = dict(sorted(distances.items(), key=operator.itemgetter(1)))
    if get_top:
        return list(distances.keys())[0]
    else:
        return list(distances.keys())[:4]

def reco_get_image_distance_between(cluster_id, centroid_difference, z_given):
    z_transformed = np.add(np.array(z_given), np.array(centroid_difference))
    df_with_cluster = product_df[product_df['cluster_id'] == cluster_id]
    if use_fixed_epsilon:
        data_ = get_nearest_image(z_transformed,
            df_with_cluster[['img_id', 'z_fixed_code_1', 'z_fixed_code_2', 'z_fixed_code_3', 'z_fixed_code_4', 
            'z_fixed_code_5', 'z_fixed_code_6', 'z_fixed_code_7', 'z_fixed_code_8', 
            'z_fixed_code_9', 'z_fixed_code_10', 'z_fixed_code_11', 'z_fixed_code_12', 
            'z_fixed_code_13', 'z_fixed_code_14', 'z_fixed_code_15', 'z_fixed_code_16']], 
            get_top=True, metrics='euclidean')

    else:
        data_ = get_nearest_image(z_transformed,
            df_with_cluster[['img_id', 'z_mean_1', 'z_mean_2', 'z_mean_3', 'z_mean_4', 
            'z_mean_5', 'z_mean_6', 'z_mean_7', 'z_mean_8', 
            'z_mean_9', 'z_mean_10', 'z_mean_11', 'z_mean_12', 
            'z_mean_13', 'z_mean_14', 'z_mean_15', 'z_mean_16',
            'z_log_var_1', 'z_log_var_2', 'z_log_var_3', 'z_log_var_4', 
            'z_log_var_5', 'z_log_var_6', 'z_log_var_7', 'z_log_var_8', 
            'z_log_var_9', 'z_log_var_10', 'z_log_var_11', 'z_log_var_12', 
            'z_log_var_13', 'z_log_var_14', 'z_log_var_15', 'z_log_var_16']],
            get_top=True, metrics='max_likelihood')
    return data_

def predict_cluster(z_given):
    predicted_cluster = -1
    highest_score = float("inf")

    for i in range(centroids_row):
        z_given = np.array(z_given)
        dist = get_distance(metric='euclidean', z_=z_given, data_vector=centroids[i])
        if dist < highest_score:
            highest_score = dist
            predicted_cluster = i
    return predicted_cluster


def get_recommendation_images(z_given):
    cluster_id = predict_cluster(z_given)
    recommendation_images = []

    for i in range(len(centroids_difference)):
        if i == cluster_id:
            continue
        else:
            recommendation_images.append(reco_get_image_distance_between(i, centroids_difference[i][j], z_given))
    return recommendation_images

def get_most_similar_images(z_given):
    cluster_id = predict_cluster(z_given)
    df_with_cluster = product_df[product_df['cluster_id'] == cluster_id]
    if use_fixed_epsilon:
        img_names = get_nearest_image(z_given,
            df_with_cluster[['img_id', 'z_fixed_code_1', 'z_fixed_code_2', 'z_fixed_code_3', 'z_fixed_code_4', 
            'z_fixed_code_5', 'z_fixed_code_6', 'z_fixed_code_7', 'z_fixed_code_8', 
            'z_fixed_code_9', 'z_fixed_code_10', 'z_fixed_code_11', 'z_fixed_code_12', 
            'z_fixed_code_13', 'z_fixed_code_14', 'z_fixed_code_15', 'z_fixed_code_16']], 
            get_top=False, metrics='euclidean')

    else:
        img_names = get_nearest_image(z_given,
            df_with_cluster[['img_id', 'z_mean_1', 'z_mean_2', 'z_mean_3', 'z_mean_4', 
            'z_mean_5', 'z_mean_6', 'z_mean_7', 'z_mean_8', 
            'z_mean_9', 'z_mean_10', 'z_mean_11', 'z_mean_12', 
            'z_mean_13', 'z_mean_14', 'z_mean_15', 'z_mean_16',
            'z_log_var_1', 'z_log_var_2', 'z_log_var_3', 'z_log_var_4', 
            'z_log_var_5', 'z_log_var_6', 'z_log_var_7', 'z_log_var_8', 
            'z_log_var_9', 'z_log_var_10', 'z_log_var_11', 'z_log_var_12', 
            'z_log_var_13', 'z_log_var_14', 'z_log_var_15', 'z_log_var_16']],
            get_top=False, metrics='max_likelihood')
    return img_names


def fake_loss(y_true, y_pred):
    reconstruction_loss = binary_crossentropy(K.flatten(y_true),
                K.flatten(y_pred))
    return reconstruction_loss

#### FUNCTIONS FOR RECOMMENDING AND SIMILARITY LOOKUP ABOVE

def home(request):
    json_list = simplejson.dumps(random_z)
    image_encoded = decode(random_z)
    context = {
        'json_list': json_list,
        'first_img': image_encoded
    }

    return render(request, 'appchemy/home.html', context)

def decode(latent_vectors):
    z_fake = np.array(latent_vectors)
    z_fake = z_fake.reshape(-1, z_dim)
    global graph
    with graph.as_default():
        decoded_output = decoder_loaded.predict(z_fake)
    
    plt.imsave(cwd + path_to_static + '/' + 'gen.jpeg', decoded_output[0])

    with open(cwd + path_to_static + '/' + 'gen.jpeg', "rb") as f:
        encoded_string = base64.b64encode(f.read())
    
    prefix = "data:image/jpeg;base64,"
    encoded_string = prefix + encoded_string.decode("utf-8") 
    
    return encoded_string

def check(request):
    z_0 = float(request.GET.get('z_0', None))
    z_1 = float(request.GET.get('z_1', None))
    z_2 = float(request.GET.get('z_2', None))
    z_3 = float(request.GET.get('z_3', None))
    z_4 = float(request.GET.get('z_4', None))
    z_5 = float(request.GET.get('z_5', None))
    z_6 = float(request.GET.get('z_6', None))
    z_7 = float(request.GET.get('z_7', None))
    z_8 = float(request.GET.get('z_8', None))
    z_9 = float(request.GET.get('z_9', None))
    z_10 = float(request.GET.get('z_10', None))
    z_11 = float(request.GET.get('z_11', None))
    z_12 = float(request.GET.get('z_12', None))
    z_13 = float(request.GET.get('z_13', None))
    z_14 = float(request.GET.get('z_14', None))
    z_15 = float(request.GET.get('z_15', None))
    
    latent_code = [z_0, z_1, z_2, z_3, z_4, z_5, z_6, z_7, z_8, z_9,
                    z_10, z_11, z_12, z_13, z_14, z_15]
    
    image_encoded = decode(latent_code)

    data = {
        'image_encoded': image_encoded
    }
    return JsonResponse(data)


def recommend(request):
    z_0 = float(request.GET.get('z_0', None))
    z_1 = float(request.GET.get('z_1', None))
    z_2 = float(request.GET.get('z_2', None))
    z_3 = float(request.GET.get('z_3', None))
    z_4 = float(request.GET.get('z_4', None))
    z_5 = float(request.GET.get('z_5', None))
    z_6 = float(request.GET.get('z_6', None))
    z_7 = float(request.GET.get('z_7', None))
    z_8 = float(request.GET.get('z_8', None))
    z_9 = float(request.GET.get('z_9', None))
    z_10 = float(request.GET.get('z_10', None))
    z_11 = float(request.GET.get('z_11', None))
    z_12 = float(request.GET.get('z_12', None))
    z_13 = float(request.GET.get('z_13', None))
    z_14 = float(request.GET.get('z_14', None))
    z_15 = float(request.GET.get('z_15', None))
    
    latent_code = [z_0, z_1, z_2, z_3, z_4, z_5, z_6, z_7, z_8, z_9,
                    z_10, z_11, z_12, z_13, z_14, z_15]

    most_similars = [str(i) + '.jpg' for i in get_most_similar_images(latent_code)]
    recommendations = [str(i) + '.jpg' for i in get_recommendation_images(latent_code)]

    data = {
        'most_similar': most_similars[0],
        'similar1': most_similars[1],
        'similar2': most_similars[2],
        'similar3': most_similars[3],
        'reco1': recommendations[0],
        'reco2': recommendations[1],
        'reco3': recommendations[2],
        'reco4': recommendations[3]
    }

    return JsonResponse(data)



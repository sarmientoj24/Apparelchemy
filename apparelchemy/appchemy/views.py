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
K.clear_session()
decoder_loaded = load_model(cwd + path_to_static + '/decoder.h5')
decoder_loaded._make_predict_function()
graph = tf.get_default_graph()

cluster_centroids = None
product_df = None
use_fixed_eps_cl = True

try:
    conn = sqlite3.connect(db_file_path)
    if use_fixed_eps_cl:
        sql = '''
            SELECT cluster_id, z_fixed_code_1, z_fixed_code_2, z_fixed_code_3, z_fixed_code_4, 
                z_fixed_code_5, z_fixed_code_6, z_fixed_code_7, z_fixed_code_8, 
                z_fixed_code_9, z_fixed_code_10, z_fixed_code_11, z_fixed_code_12, 
                z_fixed_code_13, z_fixed_code_14, z_fixed_code_15, z_fixed_code_16
            FROM centroids
            '''
    else:
        sql = '''
        SELECT cluster_id, z_mean_1, z_mean_2, z_mean_3, z_mean_4,
            z_mean_5, z_mean_6, z_mean_7, z_mean_8,
            z_mean_9, z_mean_10, z_mean_11, z_mean_12,
            z_mean_13, z_mean_14, z_mean_15, z_mean_16
        FROM centroids
        '''

    cluster_centroids = pd.read_sql_query(sql, conn)
    if use_fixed_epsilon:
        sql = '''
            SELECT img_id, cluster_id, z_fixed_code_1, z_fixed_code_2, z_fixed_code_3, z_fixed_code_4, 
                    z_fixed_code_5, z_fixed_code_6, z_fixed_code_7, z_fixed_code_8, 
                    z_fixed_code_9, z_fixed_code_10, z_fixed_code_11, z_fixed_code_12, 
                    z_fixed_code_13, z_fixed_code_14, z_fixed_code_15, z_fixed_code_16
            FROM products
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
            FROM products
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


random_zs = [[1.26676118373871, 2.21388173103333, 0.930892586708069, -0.178928971290588, 0.521288692951202, 2.14463138580322, -0.634811460971832, 1.74575090408325, 1.67183041572571, 0.0063360296189785, 1.12749469280243, -2.52685499191284, -0.0138617604970932, -0.343542039394379, -1.11246991157532, -0.076019287109375],
        [-0.171954542398453, -0.563254594802856, -0.00667597353458405, 0.687349200248718, 0.631580591201782, -0.420796066522598, -0.676108777523041, -0.400855630636215, 0.902964770793915, 2.68024015426636, 0.34944024682045, -0.107018806040287, -1.35053563117981, -0.223683953285217, -0.0077747106552124, -1.22472310066223],
        [-0.0678113102912903, -0.39513087272644, 0.114218540489674, 0.897463619709015, 0.855562508106232, 0.537120640277863, 0.796972393989563, 0.421968430280685, 1.26290559768677, 0.759354770183563, 1.90803694725037, -0.238192439079285, 0.0788260996341705, -0.648002743721008, 0.702735483646393, 0.560592710971832],
        [1.29338300228119, -0.887708365917206, 0.965926110744476, 0.555625438690186, 0.297600895166397, 0.421080350875854, 0.780524134635925, -1.77993321418762, 1.15440142154694, 0.383029580116272, 1.31792187690735, 0.890990555286407, 0.872128486633301, -1.74744272232056, -1.10591745376587, 0.915605962276459],
        [0.726071178913116, 0.474030524492264, -0.725480437278748, 0.251266211271286, -2.50118708610535, -0.0419515073299408, 0.795810639858246, -0.566275238990784, 0.135476514697075, -0.918573796749115, -0.340194046497345, 1.88385081291199, -0.0159428864717484, -0.355661898851395, -1.19611549377441, 0.138127714395523],
        [0.360172927379608, 0.0678473338484764, 0.722739994525909, 2.15070343017578, -0.932752192020416, -0.537973344326019, -1.56229412555695, -0.684384822845459, 1.86349654197693, -1.65411639213562, -1.12361991405487, -0.521878361701965, 0.32279908657074, 0.662870049476624, -0.327676475048065, 0.0612372979521751],
        [0.102847933769226, -0.54939079284668, 0.840222477912903, 0.491198033094406, -0.420244485139847, -0.750905692577362, -1.49723076820374, 0.615721583366394, 0.101508535444736, -1.97548830509186, -0.445656508207321, -0.677137017250061, 0.932354927062988, -0.719394147396088, 0.777990221977234, -0.0722629129886627],
        [-4.22261619567871, -0.429794311523437, -0.734515428543091, 0.394218146800995, 0.353307962417603, -1.40234768390656, 0.250515639781952, 0.729845702648163, -0.574816763401031, -1.14689254760742, -0.984716475009918, 0.465100198984146, -0.906986355781555, -1.54767227172852, -0.859002232551575, -1.07655143737793],
        [-0.669682860374451, -0.892655670642853, 0.466560214757919, 1.04297637939453, 1.34347748756409, 2.28552007675171, -0.723819375038147, -0.535770356655121, 0.804271042346954, -1.62950658798218, 1.33326864242554, 2.34245705604553, -0.56207811832428, 0.267478227615356, 1.15048861503601, -0.872880756855011],
        [-2.38615489006042, 0.520309209823608, 0.319325178861618, -0.205691069364548, 1.84825479984283, 1.62761831283569, 2.29895401000977, -0.280187427997589, -0.557038545608521, -0.106033459305763, 1.14095687866211, -0.702694177627563, -0.251458168029785, -1.55721127986908, 1.19954550266266, 0.80208295583725]
    ]
# random_z = random_zs[random.randint(0, 10)]
# random_z = [1.38385927677155, 0.217696711421013, -0.147801756858826, 2.06325745582581, 1.00840866565704, -1.52980017662048, -0.834733724594116, -2.05726146697998, -0.582863211631775, 0.474668443202972, -0.641297936439514, -0.60924369096756, -0.781900465488434, 0.803325593471527, 0.329163014888763, -0.615447521209717]
# random_z = [2.08627605438232, 0.107571750879288, -0.99444168806076, -0.832373559474945, 1.17749834060669, 2.75100922584534, -1.50681674480438, -2.24276733398437, -0.867893636226654, -1.23217165470123, -2.00697350502014, -0.766562223434448, 0.131717354059219, 0.421192348003387, 2.03946375846863, -0.258566498756409]
# random_z = [-0.670445621013641, -1.57144284248352, 0.935537815093994, 1.56398510932922, 2.09379816055298, -2.15937972068787, 0.6727574467659, -1.49047875404358, 1.45445048809052, -1.49117815494537, -0.633044123649597, 0.481609880924225, -2.0899031162262, 0.687041759490967, 0.0839441418647766, 0.6331427693367]
# Footwear
# random_z = [-0.313586562871933, -0.368841946125031, -3.43225955963135, 0.924136936664581, -0.4041807949543, 0.845981359481812, 1.49961996078491, 4.01226139068604, 1.30523061752319, -3.18544793128967, 1.33048987388611, -1.60870575904846, -1.52716660499573, 0.128403916954994, -2.25381469726562, -0.0326428934931755]
# Bags
#random_z = [0.0392890833318233, 0.621786177158356, 0.0462551936507225, 2.04443383216858, 2.07853055000305, 0.250094652175903, 0.482009649276733, 0.420230984687805, -0.718189835548401, -1.50380396842957, -1.07337486743927, -1.63980042934418, 2.01836967468262, 0.84935849905014, 0.0145452097058296, 0.198046207427979]
# random_z = [0.370983362197876, 0.708823084831238, 0.612207651138306, 0.513378798961639, 0.745416224002838, 0.143906086683273, 0.901776254177094, -1.37003028392792, -1.05531167984009, 0.642322063446045, -0.280464559793472, 1.35713589191437, 0.645812392234802, -0.548672199249268, 1.44595062732697, 2.00694727897644]
# Eyewear USED
# random_z = [0.479903340339661, -1.06710636615753, 0.164914309978485, 0.749682664871216, 0.155537769198418, 0.862937688827515, 1.40217542648315, 0.428536295890808, 2.06363844871521, 2.29493713378906, 0.0052320659160614, 0.574548840522766, -0.27201521396637, 0.000100739300251007, -1.45348381996155, -0.488920778036118]
#Footwear
random_z = [2.61907339096069, 1.80651700496674, 0.470471680164337, 1.71820342540741, -1.5074999332428, 0.64454174041748, -0.651311874389648, -1.23546850681305, 0.336672842502594, -0.636587679386139, 1.62876343727112, -2.89920544624329, -0.181355029344559, 0.327389568090439, -1.67370533943176, 1.1787041425705]
# random_z = [0.0674480795860291, 1.68341541290283, 1.95117652416229, 0.111782215535641, 0.0594888776540756, 1.80955505371094, 0.285866558551788, 0.861814260482788, -0.0665407329797745, -0.219140008091927, 0.900414645671844, -1.54734075069427, -0.0237731039524078, 1.08666563034058, -0.701632261276245, -0.487442195415497]
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
        return list(distances.keys())[:3]

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
        'reco1': recommendations[0],
        'reco2': recommendations[1]
    }

    return JsonResponse(data)



import ee
vCl = ee.FeatureCollection('projects/ee-wherediddavidgo/assets/s2_platte_centerlines')


def compute_otsu_threshold(histogram):
    """Identifies threshold value from series of pixel values in a certain band obtained using ee.Reducer.histogram
    such that it maximizes between sum of squares. Code adapted for python from https://medium.com/google-earth/otsus-method-for-image-segmentation-f5c48f405e.
    Original method devised by Otsu, 1979."""
    counts = ee.Array(ee.Dictionary(histogram).get('histogram'))
    means = ee.Array(ee.Dictionary(histogram).get('bucketMeans'))

    size = means.length().get([0])
    total = counts.reduce(ee.Reducer.sum(), [0]).get([0])
    sum = means.multiply(counts).reduce(ee.Reducer.sum(), [0]).get([0])
    mean = sum.divide(total)

    indices = ee.List.sequence(1, size)

    # Compute between sum of squares (BSS), where each mean partitions the data.
    def compute_bss(i):
        i = ee.Number(i)
        a_counts = counts.slice(0, 0, i)
        a_count = a_counts.reduce(ee.Reducer.sum(), [0]).get([0])
        a_means = means.slice(0, 0, i)
        a_mean = a_means.multiply(a_counts).reduce(ee.Reducer.sum(), [0]).get([0]).divide(a_count)
        
        b_count = total.subtract(a_count)
        b_mean = sum.subtract(a_count.multiply(a_mean)).divide(b_count)

        return a_count.multiply(a_mean.subtract(mean).pow(2)).add(b_count.multiply(b_mean.subtract(mean).pow(2)))
    
    bss = indices.map(compute_bss)
    # print(bss)

    return means.sort(bss).get([-1])








def compute_otsu_threshold_gpt(histogram):
    """Applies Otsu's thresholding with error handling."""
    

    
    # Extract histogram data safely
    counts, means = safe_histogram_extraction(histogram)
    
    # Ensure arrays are valid
    size = means.length().get([0])
    
    # Fallback threshold in case of empty data
    def empty_case():
        return ee.Image(-999)  # Default threshold if no valid data

    # Perform Otsu's method if data is valid
    def otsu_logic():
        total = counts.reduce(ee.Reducer.sum(), [0]).get([0])
        sum_values = means.multiply(counts).reduce(ee.Reducer.sum(), [0]).get([0])
        mean = sum_values.divide(total)

        indices = ee.List.sequence(1, size.subtract(1))

        def compute_bss(i):
            i = ee.Number(i)
            
            # Partition A
            a_counts = counts.slice(0, 0, i)
            a_count = a_counts.reduce(ee.Reducer.sum(), [0]).get([0])
            
            a_means = means.slice(0, 0, i)
            a_mean = a_means.multiply(a_counts).reduce(ee.Reducer.sum(), [0]).get([0]).divide(a_count)

            # Partition B
            b_count = total.subtract(a_count)
            b_mean = sum_values.subtract(a_count.multiply(a_mean)).divide(b_count)

            # Between-class variance
            return a_count.multiply(a_mean.subtract(mean).pow(2)).add(
                b_count.multiply(b_mean.subtract(mean).pow(2))
            )

        # Map over indices to get BSS values
        bss = indices.map(compute_bss)

        # Return threshold with max BSS
        return means.sort(bss).get([-1])

    # Apply Otsu or fallback
    threshold = ee.Algorithms.If(
        size.gt(1),
        otsu_logic(),
        empty_case()
    )

    return ee.Image(threshold)




# Function to safely extract histogram data
def safe_histogram_extraction(histogram):
    """Safely extracts histogram data with error handling."""
    
    histogram = ee.Dictionary(histogram)
    
    counts = ee.Array(histogram.get('histogram', [0]))  # Fallback to [0] if missing
    means = ee.Array(histogram.get('bucketMeans', [0]))

    # Ensure valid array lengths
    safe_counts = ee.Algorithms.If(
        counts.length().gt(0),
        counts,
        ee.Array([0])
    )

    safe_means = ee.Algorithms.If(
        means.length().gt(0),
        means,
        ee.Array([0])
    )

    return ee.Array(safe_counts), ee.Array(safe_means)


def UNPACK_SCL(scene):
    scl = scene.select('SCL')

    snowmask = scl.eq(11).rename('snow_mask')
    cloudmask = scl.eq(7).Or(scl.eq(8)).Or(scl.eq(9)).rename('cloud_mask')
    cloudwatermask = cloudmask.And(scene.select('water_mask')).rename('cloudwater_mask')

    return scene.addBands([snowmask, cloudmask, cloudwatermask])


def DN_TO_REFLECTANCE(scene):
    return scene.divide(10000).copyProperties(scene, scene.propertyNames())
    


def get_NDWI(scene):
    """Computes McFeeters MNDWI for Sentinel 2 imagery. (Green - Nir) / (Green + Nir)."""
    green = scene.select('B3')\
        .divide(10000)
    nir = scene.select('B8')\
        .divide(10000)
    
    NDWI = green.subtract(nir).divide(green.add(nir))
    return NDWI.rename('NDWI')


def ADD_WATER_MASK(scene, polygon):
def ADD_WATER_MASK(scene, polygon, dynamic=False):
    """Gets Otsu threshold for Sentinel 2 NIR band and McFeeters NDWI. Water pixels must have NIR values below the NIR threshold and MNDWI values above the MNDWI threshold.
    Intersection between NIR and MNDWI masks is the final water mask."""
    NDWI_scene = get_NDWI(scene)
    nir_scene = scene.select('B8')\
        .divide(10000)

    nir_histo = nir_scene.reduceRegion(**{'reducer': ee.Reducer.histogram(), 'geometry': polygon, 'maxPixels': 1000000000}).get('B8')
    NDWI_histo = NDWI_scene.reduceRegion(**{'reducer': ee.Reducer.histogram(), 'geometry': polygon, 'maxPixels': 1000000000}).get('NDWI')
    if dynamic:
        nir_histo = nir_scene.reduceRegion(**{'reducer': ee.Reducer.histogram(), 'geometry': polygon, 'maxPixels': 1000000000}).get('B8')
        NDWI_histo = NDWI_scene.reduceRegion(**{'reducer': ee.Reducer.histogram(), 'geometry': polygon, 'maxPixels': 1000000000}).get('NDWI')


    polygon_in_scene = ee.Algorithms.If(
        scene.geometry().intersects(polygon),
        True,
        False
    )
    nir_threshold = ee.Number(
        ee.Algorithms.If(
            polygon_in_scene,
            compute_otsu_threshold(nir_histo),
            2
        polygon_in_scene = ee.Algorithms.If(
            scene.geometry().intersects(polygon),
            True,
            False
        )
        nir_threshold = ee.Number(
            ee.Algorithms.If(
                polygon_in_scene,
                compute_otsu_threshold(nir_histo),
                2
            )
        )

    NDWI_threshold = ee.Number(
        ee.Algorithms.If(
            polygon_in_scene,
            compute_otsu_threshold(NDWI_histo),
            2
        NDWI_threshold = ee.Number(
            ee.Algorithms.If(
                polygon_in_scene,
                compute_otsu_threshold(NDWI_histo),
                2
            )
        )
    )

    if not dynamic:
        nir_threshold = 0.20
        NDWI_threshold = -0.15

    # nir_threshold = compute_otsu_threshold(nir_histo)
    # NDWI_threshold = compute_otsu_threshold(NDWI_histo)

    nir_mask = nir_scene.select('B8').lt(ee.Image(nir_threshold))
    NDWI_mask = NDWI_scene.select('NDWI').gt(ee.Image(NDWI_threshold))

    combined_mask = nir_mask.mask(NDWI_mask.mask(nir_mask)).unmask(0)\
        .rename('water_mask')

    return scene.addBands(combined_mask).set({'nirThreshold': (nir_threshold), 'ndwiThreshold': (NDWI_threshold)})



def ADD_WATER_MASK_nopoly(scene):
    """Gets Otsu threshold for Sentinel 2 NIR band and Xu MNDWI. Water pixels must have NIR values below the NIR threshold and MNDWI values above the MNDWI threshold.
    Intersection between NIR and MNDWI masks is the final water mask."""
    NDWI_scene = get_NDWI(scene)
    nir_scene = scene.select('B8')

    nir_histo = nir_scene.reduceRegion(**{'reducer': ee.Reducer.histogram(), 'maxPixels': 1000000000}).get('B8')
    NDWI_histo = NDWI_scene.reduceRegion(**{'reducer': ee.Reducer.histogram(), 'maxPixels': 1000000000}).get('NDWI')

    nir_threshold = compute_otsu_threshold(nir_histo)
    NDWI_threshold = compute_otsu_threshold(NDWI_histo)

    nir_mask = nir_scene.select('B8').lt(ee.Image(nir_threshold))
    NDWI_mask = NDWI_scene.select('NDWI').gt(ee.Image(NDWI_threshold))

    combined_mask = nir_mask.mask(NDWI_mask.mask(nir_mask)).unmask(0)\
        .rename('waterMask')

    return scene.addBands(combined_mask).set({'nirThreshold': (nir_threshold), 'ndwiThreshold': (NDWI_threshold)})



def ADD_CLOUDSHADOW_MASK(scene):
    # # Identify water pixels from the SCL band.
    # not_water = scene.select('waterMask').neq(1)

    # Identify dark NIR pixels that are not water (potential cloud shadow pixels).
    BAND_SCALE = 1e4
    NIR_DRK_THRESH = 0.15 * (2^12 -1)
    dark_pixels = scene.select('B8').lt(ee.Image(NIR_DRK_THRESH*BAND_SCALE)).rename('dark_pixels')

    # Determine the direction to project cloud shadow from clouds (assumes UTM projection).
    shadow_azimuth = ee.Number(90).subtract(ee.Number(scene.get('MEAN_SOLAR_AZIMUTH_ANGLE')))

    # Project shadows from clouds for the distance specified by the CLD_PRJ_DIST input.
    CLD_PRJ_DIST = 1
    cld_proj = (scene.select('flag_cloud').directionalDistanceTransform(shadow_azimuth, CLD_PRJ_DIST*10)
        .reproject(**{'crs': scene.select(0).projection(), 'scale': 100})
        .select('distance')
        .mask()
        .rename('cloud_transform'))

    # Identify the intersection of dark pixels with cloud shadow projection.
    shadows = cld_proj.multiply(dark_pixels).rename('flag_cloudshadow')

    # Add dark pixels, cloud projection, and identified shadows as image bands.
    return scene.addBands([dark_pixels, cld_proj, shadows])

def CALC_HILLSHADOW(scene):
    dem = ee.Image("MERIT/DEM/v1_0_3").clip(scene.geometry().buffer(9000).bounds())
    SOLAR_AZIMUTH_ANGLE = ee.Number(scene.get('MEAN_SOLAR_AZIMUTH_ANGLE'))

    SOLAR_ZENITH_ANGLE = ee.Number(90).subtract(scene.get('MEAN_SOLAR_ZENITH_ANGLE'))
    return scene.addBands(ee.Terrain.hillShadow(dem, SOLAR_AZIMUTH_ANGLE, SOLAR_ZENITH_ANGLE, 100, True).rename('flag_hillshadow'))


def ADD_SNOW_FLAG(scene):
    # using MSK_SNWPRB because it doesn't get confused by water
    msk = scene.select('MSK_SNWPRB')
    snow = msk.gt(50).rename('flag_snow')

    return scene.addBands(snow)


def TRAIN_KNN(k = 3):
    import pandas as pd
    td_table = pd.read_csv('C:/Users/dego/OneDrive - Virginia Tech/RSSA/s2_img_ids.csv')
    pids = td_table.loc[td_table['use_this_one'] == 1]['PRODUCT_ID'].to_list()
    
    training_images = [None] * len(pids)

    tp_path = 'projects/ee-wherediddavidgo/assets/knn_training_pts/'
    bois = ['B2', 'B3', 'B4', 'B8']
    classifierList = []
    monthList = []
    tileList = []

    for n in range(0, len(pids)):
        id = pids[n]
        img = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')\
            .filterMetadata('PRODUCT_ID', 'equals', id)\
            .first()
        
        tpoints_path = (tp_path + id + '_tps')
        trainingPts = ee.FeatureCollection(tpoints_path)

        imgTrainingData = img.select(bois).sampleRegions(**{
            'collection': trainingPts,
            'properties':['class'],
            'scale': 10
        })

        date = str.split(id, '_')[2]
        month = int(list(date)[4] + list(date)[5])
        monthList.append(month)

        tile = str.split(id, '_')[5]
        tileList.append(tile)

        classifier = ee.Classifier.smileKNN(k).train(**{
            'features': imgTrainingData,
            'classProperty': 'class',
            'inputProperties': bois
        })

        classifierList.append(classifier)


    return pd.DataFrame({'month': monthList, 'tile': tileList, 'classifier': classifierList})


def TRAIN_CART(minLeafPopulation = 20):
    import pandas as pd
    td_table = pd.read_csv('C:/Users/dego/OneDrive - Virginia Tech/RSSA/s2_img_ids.csv')
    pids = td_table.loc[td_table['use_this_one'] == 1]['PRODUCT_ID'].to_list()
    
    training_images = [None] * len(pids)

    tp_path = 'projects/ee-wherediddavidgo/assets/knn_training_pts/'
    bois = ['B2', 'B3', 'B4', 'B8']
    classifierList = []
    monthList = []
    tileList = []
    training_acc = []
    val_acc = []

    for n in range(0, len(pids)):
        id = pids[n]
        img = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')\
            .filterMetadata('PRODUCT_ID', 'equals', id)\
            .first()
        
        tpoints_path = (tp_path + id + '_tps')
        trainingPts = ee.FeatureCollection(tpoints_path)

        sample = img.select(bois).sampleRegions(**{
            'collection': trainingPts,
            'properties':['class'],
            'scale': 10
        })


        # Add a random value field to the sample and use it to approximately split 80%
        # of the features into a training set and 20% into a validation set.
        sample = sample.randomColumn()
        training_sample = sample.filter('random <= 0.8')
        validation_sample = sample.filter('random > 0.8')

        # # Train a CART classifier (up to 10 leaf nodes in each tree) from the
        # # training sample.
        # trained_classifier = ee.Classifier.smileCart(10).train(
        #     features=training_sample,
        #     classProperty='class',
        #     inputProperties=bois,
        # )



        date = str.split(id, '_')[2]
        month = int(list(date)[4] + list(date)[5])
        monthList.append(month)

        tile = str.split(id, '_')[5]
        tileList.append(tile)

        trained_classifier = ee.Classifier.smileCart(minLeafPopulation=20).train(**{
            'features': sample,
            'classProperty': 'class',
            'inputProperties': bois
        })

        classifierList.append(trained_classifier)

        # Get information about the trained classifier.
        result = trained_classifier.explain()

        # Get a confusion matrix and overall accuracy for the training sample.
        train_accuracy = trained_classifier.confusionMatrix()
        # display('Training error matrix', train_accuracy)
        training_acc.append(train_accuracy.accuracy())

        # Get a confusion matrix and overall accuracy for the validation sample.
        validation_sample = validation_sample.classify(trained_classifier)
        validation_accuracy = validation_sample.errorMatrix('class', 'classification')
        # display('Validation error matrix', validation_accuracy)
        val_acc.append(validation_accuracy.accuracy())


    return pd.DataFrame({'month': monthList, 'tile': tileList, 'classifier': classifierList, 'training_accuracy': training_acc, 'validation_accuracy': val_acc})

def ADD_CLOUD_MASK(scene):
    # using MSK_CLDPRB because QA60 doesn't work from Jan 2022 to Feb 2024
    msk = scene.select('MSK_CLDPRB')
    clouds = msk.gt(50).rename('flag_cloud')

    return scene.addBands(clouds)



def EXTRACT_RIVER(scene, raster_centerline, max_distance, min_island_removal):
    waterMask = scene.select('water_mask')
    bounds = waterMask.geometry()
    cl = raster_centerline\
        .select('b1')\
        .gt(0)\
        .rename('centerline_mask')\
        .clip(bounds)

    channelMask = extract_channel(waterMask, cl, max_distance)
    riverMask = remove_island(channelMask, min_island_removal)
    return scene.addBands([channelMask, riverMask])



def extract_channel(scene, raster_centerline, max_distance):
    connected_to_centerline = scene.Not().cumulativeCost(**{
        'source': raster_centerline.select('centerline_mask'),
        'maxDistance': max_distance,
        'geodeticDistance': False
    }).eq(0)

    channel = scene.updateMask(connected_to_centerline)\
        .unmask(0)\
        .updateMask(scene.gte(0))\
        .rename('channel_mask')

    return channel


def remove_island(channel, fill_size):
    fill = channel.Not()\
        .selfMask()\
        .connectedPixelCount(fill_size).lt(fill_size)
    
    river = channel.where(fill, ee.Image(1))\
        .rename(['river_mask'])

    return river




def CALCULATE_ANGLE(raster_centerline):
    w3 = (ee.Kernel.fixed(9, 9, [
        [135.0, 126.9, 116.6, 104.0, 90.0, 76.0, 63.4, 53.1, 45.0],
        [143.1, 0.0,	0.0,	0.0,	0.0,	0.0,	0.0,	0.0, 36.9],
        [153.4, 0.0,	0.0,	0.0,	0.0,	0.0,	0.0,	0.0, 26.6],
        [166.0, 0.0,	0.0,	0.0,	0.0,	0.0,	0.0,	0.0, 14.0],
        [180.0, 0.0,	0.0,	0.0,	0.0,	0.0,	0.0,	0.0, 1e-5],
        [194.0, 0.0,	0.0,	0.0,	0.0,	0.0,	0.0,	0.0, 346.0],
        [206.6, 0.0,	0.0,	0.0,	0.0,	0.0,	0.0,	0.0, 333.4],
        [216.9, 0.0,	0.0,	0.0,	0.0,	0.0,	0.0,	0.0, 323.1],
        [225.0, 233.1,  243.4,  256.0,  270.0,  284.0,  296.6,  306.9, 315.0]]))
    
    combinedReducer = ee.Reducer.sum().combine(ee.Reducer.count(), None, True)

    clAngle = (raster_centerline.mask(raster_centerline)\
      .rename(['clCleaned'])\
      .reduceNeighborhood(**{
        'reducer': combinedReducer,
        'kernel': w3,
        'inputWeight': 'kernel',
        'skipMasked': True}))

    clAngleNorm = (clAngle
      .select('clCleaned_sum')\
      .divide(clAngle.select('clCleaned_count'))\
      .mask(clAngle.select('clCleaned_count').gt(2).Not()))

    clAngleNorm = (clAngleNorm\
      .where(clAngle.select('clCleaned_count').eq(1), clAngleNorm.add(ee.Image(90))))

    return(clAngleNorm.rename(['orthDegree']))


def CALCULATE_WIDTH(scene, pts):
    crs = scene.select('B3').projection().crs()
    scale = scene.select('B3').projection().nominalScale()
    imgId = scene.get('PRODUCT_ID')
    bound = scene.select('river_mask').geometry()

    infoExport = scene.select(['river_mask', 'snow_mask', 'cloud_mask', 'cloudwater_mask'])

    infoEnds = scene.select('river_mask')

    line_stats = get_width(pts, infoExport, infoEnds, crs, scale, imgId)\
        .map(prepExport)

    return line_stats


def get_width(pts, infoExport, infoEnds, crs, scale, imgId):
    """pooop"""
    xse = pts.map(switch_to_ends)
    endStat = infoEnds.reduceRegions(**{
        'collection': xse,
        'reducer': ee.Reducer.anyNonZero()\
            .combine(ee.Reducer.count(), None, True),
        'scale': scale,
        'crs': crs
    })

    xlines = endStat.map(switch_to_line)

    combinedReducer = ee.Reducer.mean()
    xsections = infoExport.reduceRegions(**{
        'collection': xlines,
        'reducer': combinedReducer,
        'scale': scale,
        'crs': crs
    })

    xsections = xsections.map(lambda x: x.set({'img_id': imgId}))
    return xsections


def prepExport(f):

    pt_geom = ee.Geometry(f.get('longitude_latitude'))
    x = pt_geom.coordinates().get(0)
    y = pt_geom.coordinates().get(1)
                                  
    fOut = f.set({
        'width': ee.Algorithms.If(ee.Number(f.get('count')).lt(2), ee.Number(-999), ee.Number(f.get('MLength')).multiply(ee.Number(f.get('river_mask')))),
        'endsInWater': ee.Number(f.get('any')).eq(1),
        'endsOverEdge': ee.Number(f.get('count')).lt(2),
        'x': x,
        'y': y
    }).setGeometry(None)

        # .copyProperties(f, None, ['any', 'count', 'MLength', 'xc', 'yc', 'riverMask'])

    return fOut

def switch_to_line(f):
    # proj = f.geometry().projection()
    # f = f.setGeometry(ee.Geometry.LineString({'coords': [f.get('p1'), f.get('p2')], 'proj': proj, 'geodesic': False}))\
    f = f.setGeometry(ee.Geometry.LineString([f.get('p1'), f.get('p2')], None, False))\
        .set('p1', None)\
        .set('p2', None)

    return f


def switch_to_ends(f):
    f = f.setGeometry(ee.Geometry.MultiPoint([f.get('p1'), f.get('p2')]))

    return f

def poop():
    print('poop')





#### PRECOMPUTING RODEO FUNCS

# river blobs from width
def get_geoms(point):
    # blob_geom = get_blob(point)
    width = ee.Number(point.get('MLength')).divide(3)
    circle = point.geometry().buffer(width.multiply(4))
    filtered_vCl = vCl.filterBounds(circle)

    rivs_in_circle = ee.Algorithms.If(
        filtered_vCl.size().eq(0),
        False,
        True
    )

    buff = ee.Geometry.Polygon(
        ee.Algorithms.If(
            rivs_in_circle,
            filtered_vCl.union().geometry().buffer(width.multiply(3)).coordinates(),
            circle.coordinates()
        )
    )
    # edge = ee.Geometry(
    #     ee.Algorithms.If(
    #         rivs_in_circle,
    #         # ee.Geometry.MultiLineString(ee.Geometry.LinearRing(buff.coordinates().get(0)).intersection(circle).coordinates()),
    #         ee.Geometry.LinearRing(buff.coordinates().get(0)),
    #         # ee.Geometry.MultiLineString(buff.coordinates().get(0)).intersection(circle).coordinates(),
    #         ee.Geometry.LinearRing(circle.coordinates())
    #     )
    # )

    # edge = ee.Geometry.LinearRing(buff.coordinates())

    riv_length = ee.Number(
        ee.Algorithms.If(
            rivs_in_circle,
            vCl.filterBounds(buff).union().geometry().length(),
            width.multiply(8)
        )
    )
    

    return (point.set({'circle': (circle.coordinates()),
                    #    'edge': (edge.coordinates()), 
                       'buffer': (buff.coordinates()), 
                       'riv_length': riv_length,
                       'rivs_in_circle': rivs_in_circle}))
                    


def add_edge_geom(point):
    rivs_in_circle = point.get('rivs_in_circle')
    buffer_ring = ee.Geometry.LinearRing(point.get('buffer'))
    circle = ee.Geometry.LinearRing(point.get('circle'))

    edge = ee.Geometry.LinearRing(
        point.get('buffer')
    )

    return point.set('edge', edge.coordinates())
# def get_blob(point):
#     width = ee.Number(point.get('ML'))

def switch_to_blob(point):
    buffer = point.get('buffer')
    return ee.Feature(point).setGeometry(ee.Geometry.Polygon(buffer))


def switch_to_edges(point):
    edges = point.get('buffer')
    return ee.Feature(point).setGeometry(ee.Geometry.LinearRing(edges))


def switch_to_point(point):
    pt = point.get('longitude_latitude').coordinates()
    return ee.Feature(point).setGeometry(ee.Geometry.Point([pt]))



def countPixels(scene):
    pixelCount = scene.select('B3')\
        .reduceRegion(reducer=ee.Reducer.count(),
                      geometry=scene.geometry(),
                      scale=10,
                      maxPixels=1e13).get('B3')
    
    return scene.set('NUMBER_PIXELS', pixelCount)

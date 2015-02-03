import numpy as np
import pandas as pd
from scipy import sparse

def readpart(k):	

	D = pd.read_csv("data/bigeg/part-%05d.gz"%k, compression="gzip")
	D = D.fillna(0)
	
	# numeric
	N = D.ix[:,37:]
	
	# categorical
	C = pd.DataFrame(dict(
		site_us = (D['site_id']==0).astype('int').values,
		site_de = (D['site_id']==77).astype('int').values,
		site_uk = (D['site_id']==3).astype('int').values,
		site_other = (~D['site_id'].isin((0,77,3))).astype('int').values,
		# b/c2c
		b2c = (D['b2c_c2c_flag']=='B2C').astype('int').values,
		c2c = (D['b2c_c2c_flag']=='C2C').astype('int').values,
		# seller region
		sr_us = (D['bbe_seller_region']=='US').astype('int').values,
		sr_uk = (D['bbe_seller_region']=='UK').astype('int').values,
		sr_cn = (D['bbe_seller_region']=='CN').astype('int').values,
		sr_de = (D['bbe_seller_region']=='DE').astype('int').values,
		sr_au = (D['bbe_seller_region']=='AU').astype('int').values,
		sr_hk = (D['bbe_seller_region']=='HK').astype('int').values,
		sr_eu = (D['bbe_seller_region']=='EU_other').astype('int').values,
		sr_it = (D['bbe_seller_region']=='IT').astype('int').values,
		sr_ap = (D['bbe_seller_region']=='APAC').astype('int').values,
		sr_in = (D['bbe_seller_region']=='IN').astype('int').values,
		sr_ca = (D['bbe_seller_region']=='CA').astype('int').values,
		sr_th = (D['bbe_seller_region']=='TH').astype('int').values,
		sr_ao = (D['bbe_seller_region']=='APAC Other').astype('int').values,
		sr_kr = (D['bbe_seller_region']=='KR').astype('int').values,
		sr_jp = (D['bbe_seller_region']=='JP').astype('int').values,
		sr_ot = (D['bbe_seller_region']=='Other').astype('int').values,
		sr_tw = (D['bbe_seller_region']=='TW').astype('int').values,
		# css
		css_l = (D['css']=='LM').astype('int').values,
		css_m = (D['css']=='M').astype('int').values,
		css_e = (D['css']=='E').astype('int').values,
		css_r = (D['css']=='R').astype('int').values,
		css_o = (D['css']=='O').astype('int').values,
		css_n = (D['css']=='N').astype('int').values,
		# verticals 
		vert_fashion = (D['vertical_group_desc']=='Fashion').astype('int').values,
		vert_electronics = (D['vertical_group_desc']=='Electronics').astype('int').values,
		vert_homengarden = (D['vertical_group_desc']=='Home & Garden').astype('int').values,
		vert_collectibles = (D['vertical_group_desc']=='Collectibles').astype('int').values,
		vert_parts = (D['vertical_group_desc']=='Parts & Accessories').astype('int').values,
		vert_lifestyle = (D['vertical_group_desc']=='Lifestyle').astype('int').values,
		vert_media = (D['vertical_group_desc']=='Media').astype('int').values,
		vert_bizandind = (D['vertical_group_desc']=='Business & Industrial').astype('int').values,
		vert_unknown = (D['vertical_group_desc']=='Unknown').astype('int').values,
		# auction type
		auct_act = (D['auct_type_desc']=='Auction').astype('int').values,
		auct_fix = (D['auct_type_desc']=='Fixed Price').astype('int').values,
		auct_sif = (D['auct_type_desc']=='SIF').astype('int').values,
		auct_oth = (D['auct_type_desc']=='Other').astype('int').values,
		# booleans
		highvol = (D['high_vol_ind']=='Y').astype('int').values,
		lowvol = (D['low_vol_ind']=='Y').astype('int').values,
		sdnly_bad_slr = (D['sdnly_bad_slr_ind']=='Y').astype('int').values,
		stock_photo_used = (D['stock_photo_used_ind']=='Y').astype('int').values))
	
	# with open("data/bigeg/varnames.txt", 'w') as fout:
	# 	for v in ["y"] + list(C) + list(N):
	# 		fout.write("%s\n"%v)

	y = D['defect_flag'].values
	y.shape = (y.shape[0],1)
	
	yx = np.hstack( (y,C.values,N.values) )
	return( sparse.coo_matrix( yx ) )



# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 12:02:48 2024

@author: heyouwei
"""

import pandas as pd
import os
path = 'D:\\share\\code\\phylogenetic papers\\'
#filename = 'works-2024-09-30T05-38-04.csv'#phylogeny相关的论文
#filename = 'works-2024-10-23T06-10-13(AI45560).csv' #AI相关的论文
#fulltext=artificial intelligence,type=article,work=abstract available,language=EN,
#year=1900-2020
#citation count>60
#45560 results
#filename = 'works-2024-10-23T12-58-05(AI_100_27880).csv' #AI相关的论文
#fulltext=artificial intelligence,type=article,work=abstract available,language=EN,
#year=1900-2020
#citation count>100
#27880 results
filename = "works-2024-10-24T13-25-39(AI27620-1900-2022-99).csv"
#year=1900-2022
#citation count>99
#27620 results
df = pd.read_csv(path+filename,low_memory=False, on_bad_lines='skip', encoding='ISO-8859-1')
df['publication_year'] = pd.to_numeric(df['publication_year'], errors='coerce')

# 删除不能转换为整数的行
df = df.dropna(subset=['publication_year'])

# 如果需要将其转换回整数类型
df['publication_year'] = df['publication_year'].astype(int)

#%%
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
#import pygraphviz as pgv
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def getcontentedgedf(grouped_df,col_name='primary_topic.field.display_name', key_col_lst=list(range(384)), ancestor_count=3,yearwindow=5, threshold=0,minyear=1900, maxyear=2020):#生成边
    netedge_df = pd.DataFrame()
    year_lst = sorted(grouped_df[(grouped_df['publication_year'] <= maxyear)&(grouped_df['publication_year'] >= minyear)]['publication_year'].unique().tolist())
    
    for year in tqdm(year_lst,desc=f'按年生成phy_edge_df(ancestors={ancestor_count})'):
        year_df = grouped_df[grouped_df['publication_year'] == year]
        pre_year_df = grouped_df[grouped_df['publication_year'] <= year - yearwindow]
        
        if len(pre_year_df) > 0 and len(year_df) > 0:
            # 提取向量
            year_vecs = year_df[key_col_lst].values
            pre_year_vecs = pre_year_df[key_col_lst].values
            
            # 计算相似性矩阵
            similarity_matrix = cosine_similarity(year_vecs, pre_year_vecs)
            
            # 将相似性小于阈值的设为0
            similarity_matrix[similarity_matrix < threshold] = 0
            
            # 对每行选择最大的n个祖先
            for i, idx in enumerate(year_df.index):
                field = year_df.loc[idx, col_name]
                top_n_indices = np.argsort(-similarity_matrix[i])[:ancestor_count]
                
                for j in top_n_indices:
                    if similarity_matrix[i, j] > 0:  # 确保相似性大于0
                        min_id = pre_year_df.index[j]
                        dic = {
                            'year': year,
                            'field': field,
                            'preyear': grouped_df.loc[min_id, 'publication_year'],
                            'prefield': grouped_df.loc[min_id, col_name],
                            'precount': len(grouped_df[(grouped_df['publication_year'] == grouped_df.loc[min_id, 'publication_year']) & 
                                                       (grouped_df[col_name] == grouped_df.loc[min_id, col_name])]),
                            'count': len(grouped_df[(grouped_df['publication_year'] == year) & 
                                                    (grouped_df[col_name] == field)]),
                            'min_dis': (1-similarity_matrix[i, j])/2
                        }
                        netedge_df = pd.concat([netedge_df, pd.DataFrame([dic])], ignore_index=True)
    
    return netedge_df

def loadembeddingdf(path,filename='AI_abstract_df_embedding.csv'):
    abstract_df_embedding = pd.read_csv(path+filename)
    # 去除方括号并按空格分割
    split_columns = abstract_df_embedding['embedding'].str.strip('[]').str.split(expand=True)
    # 将分割后的列转换为浮点数
    split_columns = split_columns.astype(float)
    # 将分割后的列合并回原DataFrame
    abstract_df_embedding = pd.concat([abstract_df_embedding, split_columns], axis=1)
    abstract_df_embedding = abstract_df_embedding.reset_index()
    return abstract_df_embedding

def getcontentgroupdf(abstract_df_embedding,tmp_df,col_name='primary_topic.field.display_name',yearwindow=5):#根据时间窗口得到每yearwindow年的各个feild的keyword表示,abstract_df_embedding为最初的数据进行embedding后的abstract_df_embedding
    abstract_df_embedding = pd.merge(abstract_df_embedding[['id']+list(range(384))],tmp_df[['id','publication_year',col_name]],on='id',how='inner')
    abstract_df_embedding['publication_year'] = (abstract_df_embedding['publication_year'].astype(int) // yearwindow) * yearwindow
    if 'id' in abstract_df_embedding.columns:
        del abstract_df_embedding['id']
    #if 'index' in abstract_df_embedding.columns:
        #del keyword_df['index']
    abstract_grouped_df = abstract_df_embedding.groupby([col_name, 'publication_year']).mean().reset_index()
    return abstract_grouped_df

def float_to_hex_alpha(alpha):
    return hex(int(alpha * 255))[2:].zfill(2)

def rgb_to_hex(rgb):
    """
    将RGB颜色转换为十六进制颜色表示

    参数:
    rgb: tuple, 包含三个浮点数(0-1)的元组，表示RGB颜色

    返回:
    str, 十六进制颜色代码
    """
    # 将RGB值乘以255并四舍五入转换为整数
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))


def drawnet(netedge_df,col_name, field='', n=3, threshold=0, edgetype='content', yearwindow=5, setalpha=True, minyear=1875, maxyear=1950, path='D:\\share\\code\\phylogenetic papers\\'):
    link_lst = []
    alphas_lst = []
    node_colors = {}

    if field != '':
        netedge_df = netedge_df[(netedge_df['field'] == field) | (netedge_df['prefield'] == field)]
    else:
        field = 'all'
    
    unique_fields = set(netedge_df['field'].unique())|set(netedge_df['prefield'].unique())
    
    field_color_map = {f: plt.get_cmap('tab10')(i / len(unique_fields)) for i, f in enumerate(unique_fields)}

    for idx in tqdm(netedge_df.index.tolist()):
        if (netedge_df.loc[idx, 'year'] <= maxyear) and (netedge_df.loc[idx, 'year'] >= minyear):
            foc_name = str(netedge_df.loc[idx, 'year']) + '-' + str(netedge_df.loc[idx, 'year'] + yearwindow - 1) + '\n' + netedge_df.loc[idx, 'field'].split(' ')[0].rstrip(',')
            anc_name = str(netedge_df.loc[idx, 'preyear']) + '-' + str(netedge_df.loc[idx, 'preyear'] + yearwindow - 1) + '\n' + netedge_df.loc[idx, 'prefield'].split(' ')[0].rstrip(',')
            min_dis = netedge_df.loc[idx, 'min_dis']
            link_lst.append((anc_name, foc_name))
            alphas_lst.append(1 - min_dis)
            
            node_colors[foc_name] = field_color_map[netedge_df.loc[idx, 'field']]
            node_colors[anc_name] = field_color_map[netedge_df.loc[idx, 'prefield']]
    
    link_lst.reverse()
    alphas_lst.reverse()
    
    G = nx.DiGraph()
    G.add_edges_from(link_lst)

    A = nx.nx_agraph.to_agraph(G)

    for node in G.nodes():
        year = node.split('-')[0]
        A.get_node(node).attr['rank'] = year

    for node in G.nodes():
        if node in node_colors:
            A.get_node(node).attr['color'] = rgb_to_hex(node_colors[node])
            A.get_node(node).attr['penwidth'] = '5'
    if setalpha:
        for i, (u, v) in enumerate(link_lst):
            edge = A.get_edge(u, v)
            alpha_hex = float_to_hex_alpha(alphas_lst[i])
            edge.attr['color'] = f'#000000{alpha_hex}'
    else:
        for i, (u, v) in enumerate(link_lst):
            edge = A.get_edge(u, v)
            edge.attr['color'] = '#000000'
    
    A.layout(prog='dot')
    
    picname = f"{path}pics\\AI_{field}_{minyear}_{maxyear}_{edgetype}_{yearwindow}_tangled_{n}_{threshold}_tree_{edgetype}_{col_name}.png"
    A.draw(picname)

    img = plt.imread(picname)
    plt.figure(figsize=(12, 10))
    plt.imshow(img)
    plt.axis('off')

    # 添加时间流动的箭头
    img_height = img.shape[0]
    plt.text(img.shape[1] + 20, img_height // 2, 'Time', fontsize=6, color='black', va='center', rotation=90)
    plt.annotate('',
                 xy=(img.shape[1] , 50),  # 调整至较小值
                 xycoords='data',
                 xytext=(img.shape[1] , img_height ),
                 textcoords='data',
                 arrowprops=dict(arrowstyle="<-", color='black', lw=0.5, linestyle='dashed'))
    # 创建图例并移动到图片下方，调整markersize以缩小图例内容
    legend_elements = [
        Line2D([0], [0], marker='o', color='none', markeredgecolor=rgb_to_hex(field_color_map[f]), markerfacecolor='none', markersize=6, lw=2, label=f)  # 将markersize改为8
        for f in unique_fields
    ]
    ax_legend = plt.gca()  # 使用当前的坐标轴
    ax_legend.legend(handles=legend_elements, title="Academic field", loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=8)  # 底部图例，分为两列显示
    # 保存为图片
    output_picname = f"{path}pics\\AI_{field}_{minyear}_{maxyear}_{edgetype}_{yearwindow}_tangled_{n}_{threshold}_network_with_legend_{col_name}.png"
    plt.savefig(output_picname, bbox_inches='tight', dpi=300)  # 保存图像，使用高分辨率
    os.remove(picname)
    print('Save figure to '+output_picname)
    #plt.show()

    return netedge_df,plt


#%%
columns_lst = ['id','publication_year','abstract','primary_topic.subfield.display_name','primary_topic.field.display_name']#,'primary_topic.field.display_name','primary_topic.subfield.display_name']#'primary_topic.domain.display_name',
tmp_df = df[columns_lst]#[~df[columns_lst].isna()]
tmp_df = tmp_df.dropna(subset=columns_lst)
id_lst = tmp_df['id'].tolist()
# 创建字典，key 为 'primary_topic.subfield.display_name' 列，value 为 'primary_topic.field.display_name' 列
topic_dict = tmp_df.set_index('primary_topic.subfield.display_name')['primary_topic.field.display_name'].to_dict()
print(len(tmp_df))
#%%
'''
keyword_df = tmp_df[['id','publication_year', 'keywords.display_name','primary_topic.field.display_name']].copy()
keyword_df = keyword_df.set_index(['id','primary_topic.field.display_name', 'publication_year'])
keyword_df = keyword_df.drop(index='id')
keywords = keyword_df['keywords.display_name'].str.get_dummies(sep='|')
keyword_df = keyword_df.drop(columns=['keywords.display_name']).join(keywords)
key_col_lst = keywords.columns.tolist()
'''
#%%
abstract_df_embedding = loadembeddingdf(path,filename='AI_abstract_df_embedding.csv')
#abstract_df_embedding['publication_year'] = pd.to_numeric(abstract_df_embedding['publication_year'], errors='coerce')

# 删除不能转换为整数的行
#abstract_df_embedding = abstract_df_embedding.dropna(subset=['publication_year'])

# 如果需要将其转换回整数类型
#abstract_df_embedding.loc[:,'publication_year'] = abstract_df_embedding['publication_year'].astype(int)
emb_col_lst = ['id','publication_year']+list(range(384))#,'publication_year'
abstract_df_embedding_df = abstract_df_embedding[abstract_df_embedding['id'].isin(id_lst)][emb_col_lst]#.rename(columns={'publication_year':'year'})
#%%
# AI papers
ancestor_count = 3
threshold = 0.4
yearwindow=5
setalpha=True
minyear=1960
maxyear=2019
edgetype='content'
col_name = 'primary_topic.subfield.display_name'
field = 'Artificial Intelligence'#'Agricultural and Biological Sciences'
path = 'D:\\share\\code\\phylogenetic papers\\'
abstract_grouped_df = getcontentgroupdf(abstract_df_embedding_df,tmp_df,col_name,yearwindow=yearwindow)#按年聚合类似的paper
contentedge_df = getcontentedgedf(abstract_grouped_df,col_name,key_col_lst=list(range(384)),ancestor_count=ancestor_count,yearwindow=yearwindow,threshold = threshold,minyear=minyear,maxyear=maxyear)
drawnet(contentedge_df,col_name,field=field,n=ancestor_count, threshold=threshold,edgetype=edgetype,yearwindow=yearwindow,setalpha=setalpha,minyear=minyear,maxyear=maxyear,path=path)
# 保存为图片
#output_picname = f"{path}_AI_{field}_{minyear}_{maxyear}_{edgetype}_{yearwindow}_tangled_{n}_{threshold}_network_with_legend.png"
#plt.savefig(output_picname, bbox_inches='tight', dpi=300)  # 保存图像，使用高分辨率
#%%
'''
contentedge_df[(contentedge_df['prefield']=='Artificial Intelligence')&(contentedge_df['preyear']==1995)]#(contentedge_df['prefield']=='Artificial Intelligence')]
contentedge_df[(contentedge_df['prefield']=='Artificial Intelligence')].groupby('preyear')['field'].count()
contentedge_df[(contentedge_df['field']=='Artificial Intelligence')].groupby('prefield')['prefield'].count()
len(contentedge_df['field'].unique().tolist())
'''
#%%
#Validation

def generate_ancestors_dict(df):#根据df生成 ancestors_dict
    ancestors_dict = {}
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Generating ancestor dict"):
        focal_year = row['publication_year']
        #focal_field = row['field']        
        # 过滤出比 focal node 年份更早的 ancestor
        ancestors = df[df['publication_year'] < focal_year]
        ancestors_dict[index] = ancestors.index.tolist()
        #ancestor_fields = ancestors['field'].unique().tolist()
        #print(f'Node {index}: Field {focal_field} Ancestor Fields: {ancestor_fields}')
    return ancestors_dict
    
# 批量生成link_df
def generate_paperlink_df_batch(anc_count,node_df, mobileedge_df, gene_col_lst, ancestors_dict, batch_size=1):
    all_batches = []
    
    for batch in tqdm(range(batch_size), desc="Generating "+str(batch_size)+f" random networks of {anc_count} ancestors"):
        #np.random.seed(None) 
        link_df = generate_link_df(node_df, mobileedge_df, gene_col_lst, ancestors_dict)
        #link_df['brand'] = link_df['field'].apply(lambda x: topic_dict.get(x, 'Unknown'))
        #link_df['prebrand'] = link_df['prefield'].apply(lambda x: topic_dict.get(x, 'Unknown'))
        all_batches.append(link_df)
    
    return all_batches
    
#mobileedge_df
def generate_paperlink_df(node_df, mobileedge_df, gene_col_lst, ancestors_dict):#根据mobileedge_df来创建类似的随机网络
    # 创建空列表来存储生成的行
    rows = []
    
    # 使用集合来存储已生成的边以避免重复
    unique_edges = set()
    node_year_arr = node_df['year'].to_numpy()
    node_field_arr = node_df['field'].to_numpy()

    mobileedge_df['size'] = mobileedge_df.groupby(['field','year'])[['field','year']].transform('size')
    
    # 删除重复行，仅保留每个分组的一行数据
    nodeedge_df = mobileedge_df.drop_duplicates(subset=['year',  'field'])
    # 选择所需的列
    nodeedge_df = nodeedge_df[['year',  'field', 'size']]
    #print(len(nodeedge_df))
    year_arr = nodeedge_df['year'].to_numpy()
    field_arr = nodeedge_df['field'].to_numpy()
    size_arr = nodeedge_df['size'].to_numpy()
    for idx in range(len(nodeedge_df)):
        focal_year = year_arr[idx]
        focal_field = field_arr[idx]
        focal_size = size_arr[idx]
        node_idx = list(set(np.where(node_field_arr == focal_field)[0].tolist()) & set(np.where(node_year_arr == focal_year)[0].tolist()))[0]
        #node_idx = np.where(node_field_arr == focal_field)[0].tolist()[0]#node_idx是focal node在node_field_arr中的序号
        possible_ancestors = ancestors_dict.get(node_idx, [])#根据node_idx得到它的备选ancestors
        if not possible_ancestors:
            continue
        #possible_anc_fields = set(node_df[node_df.index.isin(possible_ancestors)]['field'].tolist())
        #if focal_field in possible_anc_fields:
        #print(focal_year,focal_field,len(possible_anc_fields))
        selected_ancestors = np.random.choice(
            possible_ancestors, 
            size=min(focal_size, len(possible_ancestors)), 
            replace=False
        )
         
        for ancestor_index in selected_ancestors:
            preyear = node_year_arr[ancestor_index]
            prefield = node_field_arr[ancestor_index]
            
            # 构造新的边
            new_edge = (focal_year, focal_field,  preyear, prefield)
            
            # 检查边是否已存在
            if new_edge not in unique_edges:
                unique_edges.add(new_edge)
                rows.append({
                    'year': focal_year,
                    'field': focal_field,
                    'preyear': preyear,
                    'prefield': prefield,
                })

    # 将生成的行转换为 DataFrame
    link_df = pd.DataFrame(rows, columns=['year', 'field',  'preyear', 'prefield'])
    
    return link_df

def generate_link_df(node_df, mobileedge_df, gene_col_lst, ancestors_dict): 
    # 创建空列表来存储生成的行
    rows = []
    
    # 使用集合来存储已生成的边以避免重复
    unique_edges = set()
    
    # 提取需要的数据
    node_year_arr = node_df['year'].to_numpy()
    node_field_arr = node_df['field'].to_numpy()
    node_brand_arr = node_df['brand'].to_numpy()   
    node_gene_arr = node_df[gene_col_lst].to_numpy()
    
    # 计算一次相似性矩阵，避免重复计算
    similarity_matrix = cosine_similarity(node_gene_arr)
    
    mobileedge_df['size'] = mobileedge_df.groupby(['field','year', 'brand'])[['field','year', 'brand']].transform('size')
    
    # 删除重复行，仅保留每个分组的一行数据
    nodeedge_df = mobileedge_df.drop_duplicates(subset=['year', 'brand', 'field'])
    
    # 选择所需的列
    nodeedge_df = nodeedge_df[['year', 'brand', 'field', 'size']]
    
    # 将数据转换为数组
    year_arr = nodeedge_df['year'].to_numpy()
    field_arr = nodeedge_df['field'].to_numpy()
    brand_arr = nodeedge_df['brand'].to_numpy()
    size_arr = nodeedge_df['size'].to_numpy()
    
    for idx in range(len(nodeedge_df)):
        focal_year = year_arr[idx]
        focal_field = field_arr[idx]
        focal_brand = brand_arr[idx]
        focal_size = size_arr[idx]
        node_idx = np.where(node_field_arr == focal_field)[0].tolist()[0]
        
        possible_ancestors = ancestors_dict.get(node_idx, [])
        if not possible_ancestors:
            continue
        
        selected_ancestors = np.random.choice(
            possible_ancestors, 
            size=min(focal_size, len(possible_ancestors)), 
            replace=False
        )
        
        for ancestor_index in selected_ancestors:
            preyear = node_year_arr[ancestor_index]
            prefield = node_field_arr[ancestor_index]
            prebrand = node_brand_arr[ancestor_index]
            
            # 构造新的边
            new_edge = (focal_year, focal_field, focal_brand, preyear, prefield, prebrand)
            
            # 检查边是否已存在
            if new_edge not in unique_edges:
                min_dis = (1 - similarity_matrix[idx, ancestor_index]) / 2
                unique_edges.add(new_edge)
                
                rows.append({
                    'year': focal_year,
                    'field': focal_field,
                    'brand': focal_brand,
                    'preyear': preyear,
                    'prefield': prefield,
                    'prebrand': prebrand,
                    'min_dis': min_dis
                })

    # 将生成的行转换为 DataFrame
    link_df = pd.DataFrame(rows, columns=['year', 'field', 'brand', 'preyear', 'prefield', 'prebrand', 'min_dis'])
    
    return link_df           

def compare_paperlink_ratios(all_link_dfs, mobileedge_df, comproduct_gene_df):
    # 计算 mobileedge_df 中每个 brand 的满足条件的比例
    mobileedge_df['brand_equals_prefield'] = mobileedge_df['field'] == mobileedge_df['prefield']
    mobileedge_ratios = mobileedge_df.groupby('field')['brand_equals_prefield'].mean().rename('contentedge_ratio').fillna(0)
    
    # 存储每个 link_df 的比率
    link_ratios_list = []
    
    for link_df in all_link_dfs:
        # 计算当前 link_df 中各个 brand 的满足条件的比例
        link_df['brand_equals_prefield'] = link_df['field'] == link_df['prefield']
        link_ratios = link_df.groupby('field')['brand_equals_prefield'].mean().rename('avg_link_ratio').fillna(0)
        link_ratios_list.append(link_ratios)
    
    # 将所有 link_ratios 合并到单个 DataFrame
    avg_link_ratios_df = pd.concat(link_ratios_list, axis=1).mean(axis=1).rename('avg_link_ratio').fillna(0)
    # 计算 avg_link_ratio 的均值和标准差
    avg_link_ratio_mean = avg_link_ratios_df.mean()
    avg_link_ratio_std = avg_link_ratios_df.std()
    # 计算 mobileedge_ratio 的 z-score
    mobileedge_ratios_zscore = (mobileedge_ratios - avg_link_ratio_mean) / avg_link_ratio_std
    mobileedge_ratios_zscore = mobileedge_ratios_zscore.rename('mobileedge_zscore').fillna(0)
    # 合并到最终结果中，并计算差异
    comparison_results = pd.DataFrame({
        'avg_link_ratio': avg_link_ratios_df,
        'contentedge_ratio': mobileedge_ratios,
        'mobileedge_zscore': mobileedge_ratios_zscore
    })
    
    # 计算差异
    comparison_results['avg_ratio_diff'] = comparison_results['avg_link_ratio'] - comparison_results['contentedge_ratio']
    
    # 计算 comproduct_gene_df 中每个 brand 的数量
    field_counts = comproduct_gene_df['field'].value_counts().rename('field_count')
    
    # 将数量与比较结果合并
    comparison_results = comparison_results.join(field_counts, how='left').fillna(0)
    
    # 确保所有品牌都有记录，缺失值填为 0
    comparison_results = comparison_results.fillna(0)
    
    return comparison_results[['avg_link_ratio', 'contentedge_ratio','mobileedge_zscore', 'avg_ratio_diff', 'field_count']]



def getpaperzscore(all_link_dfs, mobileedge_df,nodecount):#得到总的zscore
    #edge_size = len(mobileedge_df)
    samebrandedge_size = len(mobileedge_df[mobileedge_df['field'] == mobileedge_df['prefield']])
    exp_ratio = samebrandedge_size / nodecount#edge_size
    obs_ratio_lst = []

    for link_df in all_link_dfs:
        #link_size = len(link_df)
        samebrandlink_size = len(link_df[link_df['field'] == link_df['prefield']])
        obs_ratio = samebrandlink_size / nodecount#link_size
        obs_ratio_lst.append(obs_ratio)

    std = np.std(obs_ratio_lst)
    zscore = (exp_ratio - np.mean(obs_ratio_lst)) / std
    return zscore

#检验函数
#import numpy as np
from scipy.stats import friedmanchisquare

# 假设有 20 个样本，每个样本在 data1 和其他 99 个数据集中有对应值
#data1 = np.random.rand(20)  # data1 的样本数据
#data_list = [np.random.rand(20) for i in range(99)]  # 99 个其他数据集的数据
def friedmantest(data,randomdata_lst):
    # 将 data1 和其他数据集组合在一起
    combined_data = np.column_stack([data] + randomdata_lst)
    # 执行 Friedman 检验
    stat, p_value = friedmanchisquare(*combined_data.T)
    
    # 判断显著性并添加标志
    if p_value < 0.001:
        significance = "***"# (p < 0.001)"
    elif p_value < 0.01:
        significance = "**"# (p < 0.01)"
    elif p_value < 0.05:
        significance = "*"# (p < 0.05)"
    else:
        significance = "ns"#不显著
    
    #print(f"Friedman statistic: {stat}, p-value: {p_value} ({significance})")
    return stat,p_value,significance

def getbrandedgeratio(df,col_name='mobileedge_ratio'): #不按brand分组 
    return len(df[df['brand'] == df['prebrand']])/len(df)


def getbrandedgeratiolst(all_link_dfs,col_name='avg_link_ratio'): #不按brand分组 
    # 存储每个 link_df 的比率
    link_ratios_list = []
    for link_df in (all_link_dfs):
        link_ratios = len(link_df[link_df['brand'] == link_df['prebrand']])/len(link_df)
        link_ratios_list.append(link_ratios)
    return link_ratios_list

def getsamebrandedgeratio(df,col_name='mobileedge_ratio'):  
    df['brand_equals_prebrand'] = df['brand'] == df['prebrand']
    df = df.groupby('brand')['brand_equals_prebrand'].mean().rename(col_name).fillna(0)
    return df

def getsamebrandedgeratiolst(all_link_dfs,col_name='avg_link_ratio'): 
    # 存储每个 link_df 的比率
    link_ratios_list = []
    for link_df in (all_link_dfs):
        # 计算当前 link_df 中各个 brand 的满足条件的比例
        link_df['brand_equals_prebrand'] = link_df['brand'] == link_df['prebrand']
        link_ratios = link_df.groupby('brand')['brand_equals_prebrand'].mean().rename(col_name).fillna(0)
        link_ratios_list.append(link_ratios)
    return link_ratios_list

def permutation_test(y, y_prime, n_permutations=10000, alternative='greater'):
    observed_diff = y - np.mean(y_prime)
    combined = np.concatenate(([y], y_prime))
    perm_diffs = []

    for _ in range(n_permutations):
        np.random.shuffle(combined)
        perm_diff = combined[0] - np.mean(combined[1:])
        perm_diffs.append(perm_diff)

    # 计算p值
    if alternative == 'greater':
        p_value = np.mean(np.array(perm_diffs) >= observed_diff)
    elif alternative == 'less':
        p_value = np.mean(np.array(perm_diffs) <= observed_diff)
    else:  # 'two-sided'
        p_value = (np.sum(np.abs(perm_diffs) >= np.abs(observed_diff)) / n_permutations)

    if p_value < 0.001:
        significance = "***"
    elif p_value < 0.01:
        significance = "**"
    elif p_value < 0.05:
        significance = "*"
    elif p_value < 0.1:
        significance = "•"
    else:
        significance = "ns"
        
    return observed_diff, p_value, significance

def getgraph(df):
    G = nx.DiGraph()
    # 遍历DataFrame，创建节点和边
    for _, row in df.iterrows():
        # 定义 source 和 target 节点
        source = f"{row['preyear']}_{row['prefield']}"
        target = f"{row['year']}_{row['field']}"
        
        # 添加 source 节点，设置属性为 prebrand
        G.add_node(source, brand=row['prebrand'],year=row['preyear'],model=row['prefield'])
        
        # 添加 target 节点，设置属性为 brand
        G.add_node(target, brand=row['brand'],year=row['year'],model=row['field'] )
        
        # 添加带有 min_dis 属性的边
        G.add_edge(source, target, weight=row['min_dis'])
    return G

def gini_coefficient(values):
    """计算基尼系数"""
    n = len(values)
    if n == 0:
        return 0
    sorted_values = np.sort(values)  # 排序
    cumulative_values = np.cumsum(sorted_values)  # 累积
    gini = (n + 1 - 2 * np.sum(cumulative_values) / np.sum(sorted_values)) / n
    return gini
    
def getdegreegini(D):
    indegree_dict = dict(D.in_degree())
    outdegree_dict = dict(D.out_degree())
    indegree_values = list(indegree_dict.values())
    outdegree_values = list(outdegree_dict.values())
    indegree_gini = gini_coefficient(indegree_values)
    outdegree_gini = gini_coefficient(outdegree_values)
    return indegree_gini,outdegree_gini#,global_efficiencies.mean()
    
def getoutdegrees(D):#求每个点的outdegree
    outdegree_dict = dict(D.out_degree())
    outdegree_values = np.array(list(outdegree_dict.values()))
    return outdegree_values

def calculate_clustering(df):
    # 创建有向图
    G = nx.DiGraph()
    
    # 将数据添加到图中
    for _, row in df.iterrows():
        G.add_edge(row['prefield'], row['field'])
    
    # 计算聚类系数

    clustering_coeffs = nx.clustering(G.to_undirected())  # 使用无向图计算聚类系数

    # 计算平均聚类系数
    avg_clustering = sum(clustering_coeffs.values()) / len(clustering_coeffs) if clustering_coeffs else 0
    
    return avg_clustering

def drawdistrend(df,networktype):
    grouped_df = df.groupby('year')['min_dis'].mean().reset_index()
    # 以year为横轴画出min_dis的趋势图
    plt.plot(grouped_df['year'], grouped_df['min_dis'], marker='o')
    plt.xlabel('Year')
    plt.ylabel('Average min_dis')
    plt.title('Trend of Average min_dis Over Years of '+networktype)
    plt.xticks(range(grouped_df['year'].min(), grouped_df['year'].max() + 1, 10))
    plt.grid(True)
    plt.show()
    
def drawdiffyeartrend(df,networktype):
    df['diff_year'] = df['year']-df['preyear']
    grouped_df = df.groupby('year')['diff_year'].mean().reset_index()
    # 以year为横轴画出min_dis的趋势图
    plt.plot(grouped_df['year'], grouped_df['diff_year'], marker='o')
    plt.xlabel('Year')
    plt.ylabel('Average diff_year')
    plt.title('Trend of Average diff_year Over Years of '+networktype)
    plt.xticks(range(grouped_df['year'].min(), grouped_df['year'].max() + 1, 10))
    plt.grid(True)
    plt.show()
#%%
#按不同的ancestor数量生成phylogeny的参数初始化
threshold = -2#0.4
yearwindow=5
setalpha=True
minyear=1950
maxyear=2019
edgetype='content'
col_name = 'primary_topic.subfield.display_name'
brand_name = 'primary_topic.field.display_name'
field = ''#'Artificial Intelligence'
ancestor_count_lst = list(range(1,6,1))#从1到5
np.random.seed(42)
gene_col_lst = list(range(384))#embedding的列名
random_net_count = 100
abstract_gene_df = abstract_grouped_df.copy()#rename(columns={'publication_year':'year',col_name:'field'})
abstract_gene_df['brand'] = abstract_gene_df[col_name].apply(lambda x: topic_dict.get(x, 'Unknown'))
abstract_gene_df = abstract_gene_df[(abstract_gene_df['publication_year']<=maxyear)&(abstract_gene_df['publication_year']>=minyear)].reset_index()
abstract_df_embedding_df['publication_year'] = abstract_df_embedding_df['publication_year'].astype(int)
abstract_embedding_df = abstract_df_embedding_df[(abstract_df_embedding_df['publication_year']<=maxyear)&(abstract_df_embedding_df['publication_year']>=minyear)].reset_index()

#生成phylogeny
if 'paper_ancestors_dict' not in vars():#每个节点查找其可能的ancestors，并组成字典。用index不是id
    paper_ancestors_dict = generate_ancestors_dict(abstract_embedding_df)
if 'field_ancestors_dict' not in vars():
    field_ancestors_dict = generate_ancestors_dict(abstract_gene_df)
def getphydic(abstract_grouped_df,ancestor_count_lst,topic_dict,col_name,key_col_lst,yearwindow,threshold,maxyear): 
    contentedge_df_dic = {}
    for ancestor_count in (ancestor_count_lst):
        contentedge_df = getcontentedgedf(abstract_grouped_df,col_name,key_col_lst,ancestor_count=ancestor_count,yearwindow=yearwindow,threshold = threshold,minyear=minyear,maxyear=maxyear)
        contentedge_df['brand'] = contentedge_df['field'].apply(lambda x: topic_dict.get(x, 'Unknown'))
        contentedge_df['prebrand'] = contentedge_df['prefield'].apply(lambda x: topic_dict.get(x, 'Unknown'))
        contentedge_df_dic[ancestor_count] = contentedge_df
    return contentedge_df_dic
print('开始生成phylogeny！')
field_edge_df_dic = getphydic(abstract_gene_df,ancestor_count_lst,topic_dict,col_name,gene_col_lst,yearwindow,threshold,maxyear)
paper_gene_df = pd.merge(abstract_embedding_df,tmp_df[['id',col_name,brand_name]],on='id',how='inner').rename(columns={brand_name:'brand'})
paper_edge_df_dic = getphydic(paper_gene_df,ancestor_count_lst,topic_dict,col_name,gene_col_lst,yearwindow,threshold,maxyear)

#%%    
def getrandic(contentedge_df_dic,node_df,ancestor_count_lst,topic_dict,col_name,key_col_lst,yearwindow,threshold,maxyear,ancestors_dict=paper_ancestors_dict):
    #生成随机网络
    #field_comparison_dic = {}
    random_link_dfs_dic = {}
    for ancestor_count in ancestor_count_lst:
        contentedge_df = contentedge_df_dic[ancestor_count]    
        random_link_dfs = generate_paperlink_df_batch(ancestor_count,node_df.rename(columns={'publication_year':'year',col_name:'field'}), contentedge_df,key_col_lst ,ancestors_dict, batch_size=random_net_count)    
        random_link_dfs_dic[ancestor_count] = random_link_dfs
        #comparison_results = compare_paperlink_ratios(random_link_dfs, contentedge_df, abstract_gene_df)
        #comparison_results = comparison_results[(comparison_results[['avg_link_ratio','contentedge_ratio']] != 0).any(axis=1)].reset_index()
        #field_comparison_dic[ancestor_count] = comparison_results
    return random_link_dfs_dic
print('开始生成随机网络！')
fieldrandom_dfs_dic = getrandic(field_edge_df_dic,abstract_gene_df,ancestor_count_lst,topic_dict,col_name,gene_col_lst,yearwindow,threshold,maxyear,field_ancestors_dict)
paperrandom_dfs_dic = getrandic(paper_edge_df_dic,paper_gene_df,ancestor_count_lst,topic_dict,col_name,gene_col_lst,yearwindow,threshold,maxyear,paper_ancestors_dict)

def getoutginidic(contentedge_df_dic,random_link_dfs_dic,ancestor_count_lst):#根据contentedge_df_dic,random_link_dfs_dic求phy_outdegreegini和ran_outdegreegini_lst_dic
    # 计算density和outdegree gini
    #sig_df = pd.DataFrame()
    phy_outdegreegini_dic = {}
    ran_outdegreegini_lst_dic = {}
    for anc_count in tqdm(ancestor_count_lst):
        phy_G = getgraph(contentedge_df_dic[anc_count])
        #phy_density = nx.density(phy_G)
        _,phy_outdegreegini = getdegreegini(phy_G)  
        phy_outdegreegini_dic[anc_count] = phy_outdegreegini
        #ran_density_lst = []
        ran_outdegreegini_lst = []
        for r_i in range(random_net_count):
            ran_G = getgraph(random_link_dfs_dic[anc_count][r_i].rename(columns={'mis_dis':'min_dis'}))
            #ran_density = nx.density(ran_G)
            _,ran_outdegree = getdegreegini(ran_G)
            #ran_density_lst.append(ran_density)
            ran_outdegreegini_lst.append(ran_outdegree)
        ran_outdegreegini_lst_dic[anc_count] = np.array(ran_outdegreegini_lst)
    return phy_outdegreegini_dic,ran_outdegreegini_lst_dic
print('开始生成outdegree gini！')
field_phy_outdegreegini_dic,field_ran_outdegreegini_lst_dic = getoutginidic(field_edge_df_dic,fieldrandom_dfs_dic,ancestor_count_lst)
paper_phy_outdegreegini_dic,paper_ran_outdegreegini_lst_dic = getoutginidic(paper_edge_df_dic,paperrandom_dfs_dic,ancestor_count_lst)
#%%
#Friedman 检验 （配对，一个分布与多组分布进行比较）
#检验每个品牌中的node的ancestors中，相同brand的比率。显著大于random！
#理论：path dependdency，accumulative process
def testfriedman(contentedge_df_dic,random_link_dfs_dic,ancestor_count_lst):
    brandedgert_df = pd.DataFrame()
    for anc_count in tqdm(ancestor_count_lst):
        mobile_edgeratios = getsamebrandedgeratio(contentedge_df_dic[ancestor_count])
        random_edgeratio_lst = getsamebrandedgeratiolst(random_link_dfs_dic[anc_count])
        stat,p_value,significance = friedmantest(mobile_edgeratios,random_edgeratio_lst)
        dic = {'criterion':'1','ancestor count':anc_count,'stat':stat,'p_value':p_value,'significance':significance,'phy':mobile_edgeratios.mean(),'ran_mean':sum(r.mean() for r in random_edgeratio_lst) / len(random_edgeratio_lst)}
        brandedgert_df = pd.concat([brandedgert_df,pd.DataFrame([dic])],ignore_index=True)
    return brandedgert_df
print('进行Friedman brand检验！')
field_brandedgert_df = testfriedman(field_edge_df_dic,fieldrandom_dfs_dic,ancestor_count_lst)
paper_brandedgert_df= testfriedman(paper_edge_df_dic,paperrandom_dfs_dic,ancestor_count_lst)
field_brandedgert_df['statistic'] = field_brandedgert_df['stat'].round(4).astype(str)+' '+field_brandedgert_df['significance']
field_brandedgert_df.round(4).to_csv(path+'field_friedman.csv',index=False)
def testbrand(contentedge_df_dic,random_link_dfs_dic,ancestor_count_lst):
    brandedgert_df = pd.DataFrame()
    for anc_count in tqdm(ancestor_count_lst):
        mobile_edgeratios = getbrandedgeratio(contentedge_df_dic[ancestor_count])
        random_edgeratio_lst = getbrandedgeratiolst(random_link_dfs_dic[anc_count])
        stat,p_value,significance = permutation_test(mobile_edgeratios,random_edgeratio_lst)
        dic = {'criterion':'1','ancestor count':anc_count,'stat':stat,'p_value':p_value,'significance':significance,'phy':mobile_edgeratios,'ran_mean':sum(random_edgeratio_lst) / len(random_edgeratio_lst)}
        brandedgert_df = pd.concat([brandedgert_df,pd.DataFrame([dic])],ignore_index=True)
    return brandedgert_df
print('进行permutation brand检验！')
field_groupbrandedgert_df = testbrand(field_edge_df_dic,fieldrandom_dfs_dic,ancestor_count_lst)
paper_groupbrandedgert_df= testbrand(paper_edge_df_dic,paperrandom_dfs_dic,ancestor_count_lst)


# 2）outdegree gini network level
#outdegree gini大代表 descendent count不平均，对比random network很显著的大。理论： dominant effect，Pareto Principle，Power Law Distribution，Network Effect，Economies of Scale，Winner-Takes-All Market，Market Concentration
#社会集中化：Linzhuo, L., Lingfei, W., & James, E. (2020). Social centralization and semantic collapse: Hyperbolic embeddings of networks and text. Poetics, 78, 101428.
def testoutgini(phy_outdegreegini_dic,ran_outdegreegini_lst_dic,ancestor_count_lst,alternative='less'):
    ginitest_df = pd.DataFrame()
    for anc_count in tqdm(ancestor_count_lst):
        phylogeny_ratio = phy_outdegreegini_dic[anc_count]
        random_ratios = ran_outdegreegini_lst_dic[anc_count]
        #p_value, significance = sign_test(phylogeny_ratio, random_ratios)
        #stat,p_value, significance = wilcoxon_test(phylogeny_ratio, random_ratios)
        stat,p_value, significance = permutation_test(phylogeny_ratio, random_ratios,alternative=alternative)
        dic = {'criterion':'2','ancestor count':anc_count,'stat':stat,'p_value':p_value,'significance':significance,'phy':phylogeny_ratio,'ran_mean':random_ratios.mean()}
        ginitest_df = pd.concat([ginitest_df,pd.DataFrame([dic])],ignore_index=True)
    return ginitest_df
print('进行outgini 检验！')
field_ginitest_df = testoutgini(field_phy_outdegreegini_dic,field_ran_outdegreegini_lst_dic,ancestor_count_lst)
paper_ginitest_df = testoutgini(paper_phy_outdegreegini_dic,paper_ran_outdegreegini_lst_dic,ancestor_count_lst)

#3)cluster coefficient: innovation ecosystem
#如果显著的大，代表phylogeny比random有更多的ecosystem
def testcluster(contentedge_df_dic,random_link_dfs_dic,ancestor_count_lst,alternative='greater'):
    cluster_df = pd.DataFrame()
    for anc_count in tqdm(ancestor_count_lst):
        phy_cluster = calculate_clustering(contentedge_df_dic[anc_count])
        random_clusters = np.array([calculate_clustering(random_link_dfs_dic[anc_count][r_i]) for r_i in range(random_net_count)])
        #stat,p_value, significance = wilcoxon_test(phy_cluster, random_clusters)
        stat,p_value, significance = permutation_test(phy_cluster, random_clusters,alternative=alternative)
        #p_value, significance = sign_test(phy_cluster, random_clusters)
        dic = {'criterion':'3','ancestor count':anc_count,'stat':stat,'p_value':p_value,'significance':significance,'phy':phy_cluster,'ran_mean':random_clusters.mean()}
        cluster_df = pd.concat([cluster_df,pd.DataFrame([dic])],ignore_index=True)
    return cluster_df
print('进行cluster coefficient检验！')
field_cluster_df = testcluster(field_edge_df_dic,fieldrandom_dfs_dic,ancestor_count_lst)
paper_cluster_df = testcluster(paper_edge_df_dic,paperrandom_dfs_dic,ancestor_count_lst)

# 4） min_dis与diff_year的比较
def testmindismean(contentedge_df_dic,random_link_dfs_dic,ancestor_count_lst,alternative='less'):
    dis_mean_df = pd.DataFrame()
    for anc_count in tqdm(ancestor_count_lst):
        phy_dis_mean = contentedge_df_dic[anc_count]['min_dis'].mean()
        ran_dis_mean = np.array([random_link_dfs_dic[anc_count][r_i]['min_dis'].mean() for r_i in range(random_net_count)])
        #dis_stat,p_value, significance = wilcoxon_test(phy_dis_mean, ran_dis_mean,alternative='less' )#显著的小
        dis_stat,p_value, significance = permutation_test(phy_dis_mean, ran_dis_mean,alternative=alternative)#显著的小
        dic = {'criterion':'4.1','ancestor count':anc_count,'stat':dis_stat,'p_value':p_value,'significance':significance,'phy':phy_dis_mean,'ran_mean':ran_dis_mean.mean()}
        dis_mean_df = pd.concat([dis_mean_df,pd.DataFrame([dic])],ignore_index=True)
    return dis_mean_df

def testmindisgini(contentedge_df_dic,random_link_dfs_dic,ancestor_count_lst,alternative='greater'):
    dis_gini_df = pd.DataFrame()

    for anc_count in tqdm(ancestor_count_lst):
        phy_dis_gini = gini_coefficient(contentedge_df_dic[anc_count]['min_dis'].tolist())
        ran_dis_ginis = np.array([gini_coefficient(random_link_dfs_dic[anc_count][r_i]['min_dis'].tolist()) for r_i in range(random_net_count)])
        #p_value, significance = sign_test(phy_dis_gini, ran_dis_ginis)
        #stat,p_value, significance = wilcoxon_test(phy_dis_gini, ran_dis_ginis)
        stat,p_value, significance = permutation_test(phy_dis_gini, ran_dis_ginis,alternative=alternative)
        dic = {'criterion':'4.2','ancestor count':anc_count,'stat':stat,'p_value':p_value,'significance':significance,'phy':phy_dis_gini,'ran_mean':ran_dis_ginis.mean()}
        dis_gini_df = pd.concat([dis_gini_df,pd.DataFrame([dic])],ignore_index=True)

    return dis_gini_df
print('进行min_dis检验！')
field_dis_mean_df = testmindismean(field_edge_df_dic,fieldrandom_dfs_dic,ancestor_count_lst)
field_dis_gini_df = testmindisgini(field_edge_df_dic,fieldrandom_dfs_dic,ancestor_count_lst)
paper_dis_mean_df = testmindismean(paper_edge_df_dic,paperrandom_dfs_dic,ancestor_count_lst)
paper_dis_gini_df = testmindisgini(paper_edge_df_dic,paperrandom_dfs_dic,ancestor_count_lst)

def testdiffyearmean(contentedge_df_dic,random_link_dfs_dic,ancestor_count_lst, alternative='less'):
    diffyear_mean_df = pd.DataFrame()
    for anc_count in tqdm(ancestor_count_lst):
        phy_diffyear_mean = (contentedge_df_dic[anc_count]['year']-contentedge_df_dic[anc_count]['preyear']).mean()
        ran_diffyear_mean = np.array([(random_link_dfs_dic[anc_count][r_i]['year']-random_link_dfs_dic[anc_count][r_i]['preyear']).mean() for r_i in range(random_net_count)])
        #stat,p_value, significance = wilcoxon_test(phy_dis_mean, ran_dis_mean, alternative='less')#显著的小
        stat,p_value, significance = permutation_test(phy_diffyear_mean, ran_diffyear_mean, alternative=alternative)
        dic = {'criterion':'5.1','ancestor count':anc_count,'stat':stat,'p_value':p_value,'significance':significance,'phy':phy_diffyear_mean,'ran_mean':ran_diffyear_mean.mean()}
        diffyear_mean_df = pd.concat([diffyear_mean_df,pd.DataFrame([dic])],ignore_index=True)
    return diffyear_mean_df
     
def testdiffyeargini(contentedge_df_dic,random_link_dfs_dic,ancestor_count_lst, alternative='greater'):
    diffyear_gini_df = pd.DataFrame()
    for anc_count in tqdm(ancestor_count_lst):
        phy_diffyear_gini = gini_coefficient((contentedge_df_dic[anc_count]['year']-contentedge_df_dic[anc_count]['preyear']).tolist())
        ran_diffyear_ginis = np.array([gini_coefficient((random_link_dfs_dic[anc_count][r_i]['year']-random_link_dfs_dic[anc_count][r_i]['preyear']).tolist()) for r_i in range(random_net_count)])
        #stat,p_value, significance = wilcoxon_test(phy_diffyear_gini, ran_diffyear_ginis, alternative='less')
        stat,p_value, significance = permutation_test(phy_diffyear_gini, ran_diffyear_ginis, alternative=alternative)
        dic = {'criterion':'5.2','ancestor count':anc_count,'stat':stat,'p_value':p_value,'significance':significance,'phy':phy_diffyear_gini,'ran_mean':ran_diffyear_ginis.mean()}
        diffyear_gini_df = pd.concat([diffyear_gini_df,pd.DataFrame([dic])],ignore_index=True)
    return diffyear_gini_df
print('进行diffyear检验！')
field_diffyear_mean_df = testdiffyearmean(field_edge_df_dic,fieldrandom_dfs_dic,ancestor_count_lst)
field_diffyear_gini_df = testdiffyeargini(field_edge_df_dic,fieldrandom_dfs_dic,ancestor_count_lst)
paper_diffyear_mean_df = testdiffyearmean(paper_edge_df_dic,paperrandom_dfs_dic,ancestor_count_lst)
paper_diffyear_gini_df = testdiffyeargini(paper_edge_df_dic,paperrandom_dfs_dic,ancestor_count_lst)
#%%
drawdiffyeartrend(paper_edge_df_dic[1],'phylogeny')
drawdiffyeartrend(paperrandom_dfs_dic[1][89],'random network')
drawdistrend(field_edge_df_dic[5],'phylogeny')
drawdistrend(fieldrandom_dfs_dic[1][50],'random network')
#%%
fieldcomp_df = pd.concat([field_groupbrandedgert_df,
                          field_ginitest_df,
                          field_cluster_df,
                          field_dis_mean_df,
                          field_dis_gini_df,
                          field_diffyear_mean_df,
                          field_diffyear_gini_df
                          ],axis=0,ignore_index=True)
fieldcomp_df = fieldcomp_df.applymap(lambda x: round(x, 4) if isinstance(x, float) else x)
fieldcomp_df['statistic'] = fieldcomp_df['stat'].astype(str)+' '+fieldcomp_df['significance']
fieldcomp_df.to_csv(path+'field_testcomp_df.csv',index=False)
fieldcomp_df
#%%
def getpivot(df,col_lst=['statistic', 'phy', 'ran_mean']):
    df = df.astype(str)
    pivot_df = df.pivot_table(index='criterion', columns='ancestor count', values=col_lst, aggfunc=lambda x: ' '.join(x))
    
    # 重命名列
    pivot_df.columns = pd.MultiIndex.from_tuples([(f'ancestor_count_{col[1]}',col[0] ) for col in pivot_df.columns])
    pivot_df = pivot_df.sort_index(axis=1, level=0)
    return pivot_df
pivot_df = getpivot(fieldcomp_df,['statistic'])    
pivot_df.to_csv(path+'field_pivot_df.csv')
pivot_df = getpivot(fieldcomp_df,['phy', 'ran_mean'])    
pivot_df.to_csv(path+'mean_field_pivot_df.csv')
#%%
from math import pi

def plot_radar(df,title,color="grey"):
    # 获取唯一的 ancestor count 和 criteria
    ancestor_counts = df['ancestor count'].unique()
    criteria = df['criterion'].unique()

    # 定义颜色
    colors = ['b', 'r']
    light_green = '#90EE90'  # 浅绿色
    # 创建图形和子图
    fig, axes = plt.subplots(1, len(ancestor_counts), figsize=(20, 8), subplot_kw={'polar': True})

    if len(ancestor_counts) == 1:
        axes = [axes]  # 保持axes为列表，便于后续处理
        # 添加总的标题
    fig.suptitle(title, size=20, y=.8)
    # 循环每个 ancestor count 绘制雷达图
    for ax, count in zip(axes, ancestor_counts):
        # 过滤当前 ancestor count 的数据
        data = df[df['ancestor count'] == count]

        # 变量数目
        num_vars = len(criteria)

        # 计算每个轴的角度
        angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
        angles += angles[:1]  # 闭合

        # 画每个轴并添加标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(criteria)

        # 画 y-label
        ax.set_rlabel_position(0)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8])
        ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8"], color=color, size=7)
        ax.set_ylim(0, 1)

        # 绘制 phy 数据
        values = data['phy'].tolist()
        values += values[:1]
        ax.plot(angles, values, color=colors[0], linewidth=2, linestyle='solid', label='Phylogeny')
        ax.fill(angles, values, color=colors[0], alpha=0.4)

        # 绘制 ran_mean 数据
        values = data['ran_mean'].tolist()
        values += values[:1]
        ax.plot(angles, values, color=colors[1], linewidth=2, linestyle='solid', label='Random')
        ax.fill(angles, values, color=colors[1], alpha=0.4)
        # 设置圆圈颜色为绿色
        for spine in ax.spines.values():
            spine.set_color(color)
        # 设置网格线颜色为绿色
        ax.yaxis.grid(True, color=light_green)
        ax.xaxis.grid(True, color=light_green)
        # 添加标题
        ax.set_title(f'Ancestor Count: {count}', size=15, color='black', y=1.1)

        # 添加图例
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

    # 调整布局
    plt.tight_layout()
    plt.savefig(path+'paper_rada.png', bbox_inches='tight', dpi=300)
    plt.show()


plot_radar(fieldcomp_df,'b. Paper data','green')
#%%
# 绘制折线图
plt.figure(figsize=(10, 6))
for crit in fieldcomp_df['Criterion'].unique():
    plot_df = fieldcomp_df[fieldcomp_df['Criterion']==crit]
    ancestor_counts = plot_df['Ancestor count']
    phylogeny_means = plot_df['phy']
    random_network_means = plot_df['ran_mean']
    plt.plot(ancestor_counts, phylogeny_means, label=f'{crit} - Phylogeny', marker='o')
    plt.plot(ancestor_counts, random_network_means, label=f'{crit} - Random Network', marker='x')

plt.xlabel('Ancestor Count')
plt.ylabel('Mean Value')
plt.title('Comparison of Phylogeny and Random Network')
plt.legend()
plt.grid(True)
plt.show()


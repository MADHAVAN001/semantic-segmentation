import json
def getSuperClass(path):
    json_file = open(path)
    data = json.load(json_file)
    data = data['categories']
    supercat_names = {}
    supercat_ids = {}
    class_superclass_map_name = {}
    class_superclass_map_id = {}
    lencat = len(data)
    for x in range(lencat):
        supercat = data[x]['supercategory']
        if not supercat in supercat_names.keys():
            supercat_names[supercat] = []
            supercat_ids[supercat] = []
        supercat_names[supercat].append(data[x]['name'])
        supercat_ids[supercat].append(data[x]['id'])
        class_superclass_map_name[data[x]['name']] = supercat
        class_superclass_map_id[data[x]['id']] = supercat
    final_list_supercat = list(supercat_names.keys())
    final_list_supercat.sort()
    idxBG = final_list_supercat.index('background')
    tmp = final_list_supercat[idxBG]
    final_list_supercat[idxBG] = final_list_supercat[0]
    final_list_supercat[0] = tmp
    #Override for now due to bug in code to generate mask. Once fixed, below line can be removed
    #TODO: Run regeneration of mask and then remove below line 
    final_list_supercat = ['background', 'appliance', 'electronic', 'accessory', 'kitchen', 'sports', 'vehicle', 'furniture', 'food', 'outdoor', 'indoor', 'animal', 'person']
    return final_list_supercat

'''
getSuperClass("/home/shravan/classMaps.json")
'''


def class_id_to_name(id):
    class_to_name = {
        0: 'background',
        1: 'bottle',
        2: 'bowl',
        3: 'camera',
        4: 'can',
        5: 'laptop',
        6: 'mug',
        7: 'rubikscube'
    }
    return class_to_name[id]

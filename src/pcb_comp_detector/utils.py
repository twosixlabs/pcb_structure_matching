from detectron2.data import Metadata
from detectron2.utils.visualizer import Visualizer
from detectron2.structures import Boxes
import inspect

def draw_detections(image, detections, classes, scale=1):
    metadata = Metadata()
    metadata.set(thing_classes=classes)
    return Visualizer(image, metadata, scale=scale).draw_instance_predictions(detections).get_image()

def add_func_args_to_parser(parser, func):
    argspec = inspect.getfullargspec(func)
    print(argspec)
    for name in argspec.kwonlyargs:
        if not "--" + name in parser._optionals._option_string_actions.keys():
            if not name in argspec.annotations:
                raise Exception("Function must have type annotations")
            parser.add_argument("--" + name, default=argspec.kwonlydefaults[name], type=argspec.annotations[name])

    return set(argspec.kwonlyargs)

def instances_to_dict(instances):
    # can take an instances or list of instances
    islist = isinstance(instances, list)

    if not islist:
        instances = [instances]

    for i in range(len(instances)):
        instances[i] = instances[i].get_fields()
        for field in instances[i].keys():
            if isinstance(instances[i][field], Boxes):
                instances[i][field] = instances[i][field].tensor
            instances[i][field] = instances[i][field].cpu().tolist()

    if not islist:
        instances = instances[0]
    return instances
# Explainable-Object-induced-Action-Decision-for-Autonomous-Vehicles-for-ML-Reproducibility-Challenge

## Summary

Deep learning has had significant contributions in autonomous driving with two
major approaches. Ones such method is an end to end system utilizing the global
features found in the entire scene to make predictions. However, a large dataset
is required for accurate implementation and the black box mechanism from this
method gives no reasoning for the prediction. Another method is called pipeline
approach, which perceives local features such as the objects and obstacles around
the vehicle and then predicts the action.

The paper of interest incorporates these two approaches and involves both global
features and local features. The objects that have strong causality with vehicle
movement, are called action-inducing objects. Each action-inducing object is as-
sociated with an explanation of action change, thereby the action of the vehicle
will be explainable and the model can be sufficiently optimized based on the
explanation. Four actions (move forward, slow/stop, turn left, and turn right)
and 21 explanations are included. Besides, the result of each sample can have
multi-label since more options are available (e.g. turn left or stop). Both action
and explanation contribute to the loss function in this system.

The dataset used is the BDD100k Dataset comprising of only complicated scenes
that were manually annotated with the appropriate explanations. The training
process uses the Faster-RCNN (Mask-RCNN in the original implementation)
as the backbone to extract features from the images. These features are then
trained by a global module as well as a local branch. In the global branch, the
spatial context such as the location of action-inducing objects and the scene
context are provided using two convolutional layers. In the local branch, each
action-inducing object will be evaluated as a score and the top k objects will be
selected. Finally, the predicted action and explanation are produced based on
these two branches.

By experimenting different model architectures with only local branch, only
global branch, both, and selecting top-k action-inducing objects in the local
branch, the results show that the vehicle can have better performance in all
the actions when considering both global and local features and select top-
objects. The reason behind this is the action of the vehicle needs the spatial
information of action-inducing objects from global features and explicit detailed
context from local features.

The the proposed method is shown to successfully detect action-inducing objects
and make correct actions. One thing that can be improved is the small number
of explanations. Reality and decision making during the activity of driving is
far more complex and requires more extensive explanations. Therefore, more
explanation should be added to the system to improve performance.

## Related Background

The investigated literature utilizes two mainstream network models to act as
baseline models for the evaluation of the proposed explainable autonomous driv-
ing network. The primarily baseline model is the Resnet-50 model, a mature
neural network model used extensively in academia and industry[ 1 ]. A found
network that broke records in 2015 by achieving 3.57% error on the ImageNet
test set, due to its depth and use of recurrent neural networks to combat the van-
ishing gradient problem. The other baseline model implemented used an object-
oriented approach to autonomous driving that utilized end-to-end learning and
object instances[ 2 ]. This model is famously known for it’s outperformance of an
object-agnostic model in a Grand Theft Auto V simulator. Both of these models
are representative of the standard for autonomous driving and serve as good
baselines for evaluating the developed model.

End-to-End learning in an autonomous driving application is the ability for
an algorithm to learn how to steer the vehicle from the raw pixel values obtained
from cameras. This allows the system to train itself on minimal data from hu-
mans and are considered to have no form of reasoning. Autonomous vehicle
researchers have taken advantage of this method to create algorithms with at-
tention to detail of both context and object features.[ 2 ]However, many examples
in literature are trained in driving simulations[ 2 ], and it is known that models
trained on synthetic data typically underperform in real world applications. The
targeted paper incorporated end-to-end learning into their system that will be
fully-trained on real-life examples. The targeted model will leverage both object
detection and contextual reasoning offered by end-to-end learning from real-life
datasets.

Global and local feature and thier relationship within a scene has been an
important component for detection of objects of interest in an image. However,
there is a significant lack of attention of contextual representation in autonomous
driving. In several instances, literature has focused on either global features or
local features, but never both features and their relationship.[ 3 ] However, the
investigated paper does just that, accounting for both local and global objects
by using explanations, and how their relationship influences separation between
action-inducing objects and unimportant objects.

Attention mechanisms gives researchers the opportunity to “see what CNNs
sees” and evaluate what features the networks are targeting. In previous litera-
ture, extensive research had been conducted to show networks that focused on
a range of available data. From objects in the image, pixels that heavily collab-
orated to the prediction, to what scene’s capture the driver’s gaze.[ 4 ]However,
the investigated literature emphasizes a focus on the explanation of the scene
and provides a unique attention mechanism mapping to the autonomous driving
scene.

Previous literature has incorporated the use of explanations in autonomous
driving decisions. Several examples have been shown that use textual generation
to explain the actions of the model.[ 4 ] However, no previous research has imple-
mented action-inducing object based one-hot encoding of explanations. This is
unique as it removes the ambiguity of textual generated explanations in favor of
simple action predictions.

There are currently a substantial number of datasets available for autonomous
driving applications. Many of them are richly annotated and offer a good variabil-
ity from scene to scene. However, none of the available datasets include the detec-
tion of action-inducing objects, specifically needed for this literature. Therefore,
the researchers derived from BDD100K, an autonomous driving dataset, that
includes an extensive number of new annotations concerning action-inducing
objects found in the driving scenes. [ 5 ]

## 3 What have we done so far?

We have downloaded and examined the provided datasets, which seem to align
with the description in the paper. The dataset is a variation of the original
BDD100k where the explanation of each image is hand-labeled as a one-hot vec-
tor. The backbone model used in the paper is Mask-RCNN, which is no longer
supported by the Facebook AI team. We have attempted to set up the Mask-
RCNN on both local machine and Google Collab but neither succeeded due
to missing dependencies and broken docker files. The original Mask-RCNN has
officially been renamed to Detectron2, which is a newer framework and fully
supported. As a result, we have successfully set up Detectron2 along with all the
required dependencies on both our local machine and Google Collab. We have
tested simple object detection models pretrained by Detectron2 and the results
are very promising. We have also uploaded an image from a previous assignment
to see if the pretrained model can detect objects presented in the image and the
results showed that all objects of interest are accurately found and recognized
with a reasonable confidence. As shown inimage 1, the detected objects are also
almost perfectly segmented and separate from each other. The performance of
the model indicates great capability of the Detectron2 framework. Therefore, we
have decided to migrate the provided code to fully support Detectron2 and reproduce 
the results presented in the paper. Although the results are expected to
be the same, it is likely that there will be improvement upon successful migration
since Detectron2 is a newer framework. Some changes and compatibility notes
are documented on the official website [ 6 ]. No credit will be taken for any im-
provement in the performance caused by the improved capability of Detectron
over Mask-RCNN. However, we are also aiming to further explore the potential
capability of the proposed technique and test its limitations.
Following the strategy, we have identified some immediately obvious changes
required to make to adapt Detectron2. Here are a few examples:

- maskrcnnbenchmark.utils.miscellaneous→removed in detectron
- maskrcnnbenchmark.utils.checkpoint→detectron2.checkpoint
- maskrcnnbenchmark.modeling.detector.builddetectionmodel→
    detectron2.modeling.buildmodel

Other changes that may or may not be relevant are documented in [ 6 ]

In terms of the dataset, we have set up a bucket on Amazon AWS S3 where we
have uploaded the entire dataset. We have also established a connection between
Google Colab and the bucket to directly interact with the data. The next step
is to extract labels and action explanations from the .JSON files since both are
one-hot encoded in the BDDOIA dataset. Meanwhile, we are also prioritizing
the migration work on our local machine, which is preferred since the original
framework was implemented to run locally.


## References

1. Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning
    for image recognition. In Proceedings of the IEEE conference on computer vision
    and pattern recognition, pages 770–778, 2016.
2. Wang, D., Devin, C., Cai, Q., Yu, F., amp; Darrell, T. (2019). Deep Object-Centric
    Policies for Autonomous Driving. 2019 International Conference on Robotics and
    Automation (ICRA). doi:10.1109/icra.2019.
3. Kim, J., Rohrbach, A., Darrell, T., Canny, J., amp; Akata, Z. (2018). Textual Ex-
    planations for Self-Driving Vehicles. Computer Vision – ECCV 2018 Lecture Notes
    in Computer Science, 577-593. doi:10.1007/978-3-030-01216-8 35
4. Xia, Y., Zhang, D., Kim, J., Nakayama, K., Zipser, K., amp; Whitney, D. (2019).
    Predicting Driver Attention in Critical Situations. Computer Vision – ACCV 2018
    Lecture Notes in Computer Science, 658-674. doi:10.1007/978-3-030-20873-8 42
5. Yu, F., Chen, H., Wang, X., Xian, W., Chen, Y., Liu, F.,... Darrell, T. (2020).
    BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning. 2020
    IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).
    doi:10.1109/cvpr42600.2020.
6. https://detectron2.readthedocs.io/notes/compatibility.html



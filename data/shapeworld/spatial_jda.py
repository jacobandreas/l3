from shapeworld.dataset import CaptionAgreementDataset
from shapeworld.generators import GenericGenerator
from shapeworld.captioners import AttributesTypeCaptioner, SpatialRelationCaptioner, ExistentialCaptioner


class SpatialJdaDataset(CaptionAgreementDataset):

    dataset_name = 'spatial_jda'

    def __init__(self, validation_combinations, test_combinations, caption_size, words, language=None):
        world_generator = GenericGenerator(
            entity_counts=[4, 5],
            collision_tolerance=0.0,
            boundary_tolerance=0.0,
            validation_combinations=validation_combinations,
            test_combinations=test_combinations,
            max_provoke_collision_rate=0.0
        )
        world_captioner = ExistentialCaptioner(
            restrictor_captioner=AttributesTypeCaptioner(
                trivial_acceptance_rate=1.0
            ),
            body_captioner=SpatialRelationCaptioner(
                reference_captioner=AttributesTypeCaptioner(),
                relations=('x-rel', 'y-rel')
            )
        )
        super(SpatialJdaDataset, self).__init__(
            world_generator=world_generator,
            world_captioner=world_captioner,
            caption_size=caption_size,
            words=words,
            language=language
        )


dataset = SpatialJdaDataset
SpatialJdaDataset.default_config = dict(
    #validation_combinations=[['square', 'red', 'solid'], ['triangle', 'green', 'solid'], ['circle', 'blue', 'solid']],
    #test_combinations=[['rectangle', 'yellow', 'solid'], ['cross', 'magenta', 'solid'], ['ellipse', 'cyan', 'solid']],
    validation_combinations=None,
    test_combinations=None,
    caption_size=12,
    words=['.', 'a', 'above', 'an', 'below', 'black', 'blue', 'circle', 'cross', 'cyan', 'ellipse', 'green', 'is', 'left', 'magenta', 'of', 'pentagon', 'rectangle', 'red', 'right', 'semicircle', 'shape', 'square', 'the', 'to', 'triangle', 'white', 'yellow']
)

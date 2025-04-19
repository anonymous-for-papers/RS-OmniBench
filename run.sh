CUDA_VISIBLE_DEVICES=4 \
/data/envs/geochat/bin/python run.py \
    --data image_level_image_captioning_image_wide \
      image_level_image_classification_image_wide \
      image_level_image_vqa_image_wide \
      image_level_image_retrieval_image_wide \
      data region_level_image_od_image_wide \
      region_level_image_vg_image_wide \
      pixel_level_image_segmentation_image_wide \
      image_level_image_captioning_region_specific \
      image_level_image_classification_region_specific \
      image_level_image_vqa_region_specific \
      region_level_image_vg_region_specific \
      region_level_image_od_region_specific \
      pixel_level_image_segmentation_region_specific \
    --model GeoChat \
    --verbose \
    --mode infer \
    --reuse
    #over

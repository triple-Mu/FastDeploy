// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "fastdeploy/vision/ocr/ppocr/det_preprocessor.h"

#include "fastdeploy/vision/ocr/ppocr/utils/ocr_utils.h"

namespace fastdeploy {
namespace vision {
namespace ocr {

std::pair<int, int> cal_dst_size(int src_height, int src_width, int long_min,
                                 int short_min, int base = 32) {
  float base_ratio = (float)long_min / (float)short_min;

  float cur_ratio;
  int dst_height;
  int dst_width;
  int cur_long;
  int cur_short;
  bool swap;

  if (src_height > src_width) {
    cur_long = src_height;
    cur_short = src_width;
    cur_ratio = (float)cur_long / (float)cur_short;
    swap = false;
  } else {
    cur_long = src_width;
    cur_short = src_height;
    cur_ratio = (float)cur_long / (float)cur_short;
    swap = true;
  }

  if (cur_long > long_min && cur_short > short_min) {
    dst_height = swap ? cur_short : cur_long;
    dst_width = swap ? cur_long : cur_short;
    dst_height = (dst_height + base - 1) / base * base;
    dst_width = (dst_width + base - 1) / base * base;
    dst_height = std::max(dst_height, base);
    dst_width = std::max(dst_width, base);
    return std::make_pair(dst_height, dst_width);
  }
  if (cur_ratio > base_ratio) {
    float ratio = (float)short_min / (float)cur_short;
    int new_long = std::ceil(ratio * (float)cur_long);
    dst_height = swap ? short_min : new_long;
    dst_width = swap ? new_long : short_min;
  } else {
    float ratio = (float)long_min / (float)cur_long;
    int new_short = std::ceil(ratio * (float)cur_short);
    dst_height = swap ? new_short : long_min;
    dst_width = swap ? long_min : new_short;
  }
  dst_height = (dst_height + base - 1) / base * base;
  dst_width = (dst_width + base - 1) / base * base;
  dst_height = std::max(dst_height, base);
  dst_width = std::max(dst_width, base);
  return std::make_pair(dst_height, dst_width);
}

std::array<int, 4> DBDetectorPreprocessor::OcrDetectorGetInfo(
    FDMat* img, int max_size_len) {
  int w = img->Width();
  int h = img->Height();
  if (static_shape_infer_) {
    return {w, h, det_image_shape_[2], det_image_shape_[1]};
  }

  float ratio = 1.f;
  int max_wh = w >= h ? w : h;
  if (max_wh > max_size_len) {
    if (h > w) {
      ratio = float(max_size_len) / float(h);
    } else {
      ratio = float(max_size_len) / float(w);
    }
  }
  int resize_h = int(float(h) * ratio);
  int resize_w = int(float(w) * ratio);
  resize_h = std::max(int(std::round(float(resize_h) / 32) * 32), 32);
  resize_w = std::max(int(std::round(float(resize_w) / 32) * 32), 32);

  return {w, h, resize_w, resize_h};
  /*
   *ratio_h = float(resize_h) / float(h);
   *ratio_w = float(resize_w) / float(w);
   */
}

std::array<int, 4> DBDetectorPreprocessor::OcrDetectorGetInfo(FDMat* img) {
  int w = img->Width();
  int h = img->Height();
  if (static_shape_infer_) {
    return {w, h, det_image_shape_[2], det_image_shape_[1]};
  }
  std::pair<int, int> dst_size =
      cal_dst_size(h, w, longside_size_, shortside_size_);
  // std::cout << w << " * " << h << " -> " << dst_size.second << " * "
  //           << dst_size.first << std::endl;
  return {w, h, dst_size.second, dst_size.first};
}

DBDetectorPreprocessor::DBDetectorPreprocessor() {
  resize_op_ = std::make_shared<Resize>(-1, -1);

  std::vector<float> value = {0, 0, 0};
  pad_op_ = std::make_shared<Pad>(0, 0, 0, 0, value);

  normalize_permute_op_ = std::make_shared<NormalizeAndPermute>(
      std::vector<float>({0.485f, 0.456f, 0.406f}),
      std::vector<float>({0.229f, 0.224f, 0.225f}), true);
}

bool DBDetectorPreprocessor::ResizeImage(FDMat* img, int resize_w, int resize_h,
                                         int max_resize_w, int max_resize_h) {
  resize_op_->SetWidthAndHeight(resize_w, resize_h);
  (*resize_op_)(img);

  pad_op_->SetPaddingSize(0, max_resize_h - resize_h, 0,
                          max_resize_w - resize_w);
  (*pad_op_)(img);
  return true;
}

bool DBDetectorPreprocessor::Apply(FDMatBatch* image_batch,
                                   std::vector<FDTensor>* outputs) {
  int max_resize_w = 0;
  int max_resize_h = 0;
  batch_det_img_info_.clear();
  batch_det_img_info_.resize(image_batch->mats->size());
  for (size_t i = 0; i < image_batch->mats->size(); ++i) {
    FDMat* mat = &(image_batch->mats->at(i));
    // batch_det_img_info_[i] = OcrDetectorGetInfo(mat, max_side_len_);
    batch_det_img_info_[i] = OcrDetectorGetInfo(mat);
    max_resize_w = std::max(max_resize_w, batch_det_img_info_[i][2]);
    max_resize_h = std::max(max_resize_h, batch_det_img_info_[i][3]);
  }
  for (size_t i = 0; i < image_batch->mats->size(); ++i) {
    FDMat* mat = &(image_batch->mats->at(i));
    ResizeImage(mat, batch_det_img_info_[i][2], batch_det_img_info_[i][3],
                max_resize_w, max_resize_h);
  }

  if (!disable_normalize_ && !disable_permute_) {
    (*normalize_permute_op_)(image_batch);
  }

  outputs->resize(1);
  FDTensor* tensor = image_batch->Tensor();
  (*outputs)[0].SetExternalData(tensor->Shape(), tensor->Dtype(),
                                tensor->Data(), tensor->device,
                                tensor->device_id);
  return true;
}

}  // namespace ocr
}  // namespace vision
}  // namespace fastdeploy

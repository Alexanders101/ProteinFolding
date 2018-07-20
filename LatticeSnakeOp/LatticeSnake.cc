#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

#include "tensorflow/core/framework/shape_inference.h"

#include "LatticeSnake.h"

using namespace tensorflow;

REGISTER_OP("LatticeSnake")
        .Input("acids: float32")
        .Input("mask: bool")
        .Input("idx: int32")
        .Attr("max_length: int >= 1")
        .Attr("window_size: int >= 1")
        .Output("lattice: float32")
        .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
            int max_length, window_size;
            c->GetAttr("max_length", &max_length);
            c->GetAttr("window_size", &window_size);

            c->set_output(0, c->MakeShape({max_length, window_size, window_size, window_size}));
            return Status::OK();
        });

class LatticeSnakeOp : public OpKernel {
public:
    explicit LatticeSnakeOp(OpKernelConstruction *context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("max_length", &max_length_));
        OP_REQUIRES_OK(context, context->GetAttr("window_size", &window_size_));
    }

    void Compute(OpKernelContext *context) override {
        const Tensor &acids_T = context->input(0);
        const Tensor &mask_T = context->input(1);
        const Tensor &idx_T = context->input(2);

        OP_REQUIRES(context, TensorShapeUtils::IsVector(acids_T.shape()),
                    errors::InvalidArgument("Acid string must be of shape (N)"));

        OP_REQUIRES(context, TensorShapeUtils::IsVector(mask_T.shape()),
                    errors::InvalidArgument("Mask must be of shape (N)"));

        OP_REQUIRES(context, TensorShapeUtils::IsMatrix(idx_T.shape()),
                    errors::InvalidArgument("Lattice Snake Indices must be (N, 3)"));

        OP_REQUIRES(context, idx_T.shape().dim_size(1) == 3,
                    errors::InvalidArgument("Lattice Snake Indices must be (N, 3)"));

//        OP_REQUIRES(context, idx_T.dim_size(0) == acids_T.dim_size(0) == mask_T.dim_size(0),
//                    errors::InvalidArgument("All three inputs must have the same timestep dimensions."));

        // Cast inputs into Eigen Tensors
        const auto acids = acids_T.flat<float>();
        const auto mask = mask_T.flat<bool>();
        using IDX_TYPE = Eigen::Matrix<int, Eigen::Dynamic, 3, Eigen::RowMajor>;
        const IDX_TYPE idx = Eigen::Map<const IDX_TYPE>(
                idx_T.flat<int>().data(),
                idx_T.dim_size(0),
                idx_T.dim_size(1));

        const auto combined_N = acids.size();
        const auto N = (combined_N / 2) + 1;

        // Create output buffer
        TensorShape out_shape{max_length_, window_size_, window_size_, window_size_};
        Tensor *output_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output_tensor));
        auto output = output_tensor->tensor<float, 4>();
        output.setZero();

        // /////////////////////////////////////
        // initialize sparse array
        // /////////////////////////////////////
        using coord_t = const std::array<int, 3>;
        std::map<coord_t, float> sparse_map;

        // Acids
        for (size_t i = 0; mask(i) && i < N; ++i) {
            const std::array<int, 3> coordinate{idx(i, 0), idx(i, 1), idx(i, 2)};
            sparse_map[coordinate] = acids(i);
        }

        // Intermediate Points
        for (auto i = N; mask(i) && i < combined_N; ++i) {
            const std::array<int, 3> coordinate{idx(i, 0), idx(i, 1), idx(i, 2)};
            sparse_map[coordinate] = acids(i);
        }

        // /////////////////////////////////////
        // Fill in lattice snake
        // /////////////////////////////////////
        const int offset_size = window_size_ / 2;


        // Acids
        for (size_t timestep = 0; mask(timestep) && timestep < N; ++timestep) {
            const auto center = idx.row(timestep);

            for (int x = -offset_size; x <= offset_size; ++x) {
                for (int y = -offset_size; y <= offset_size; ++y) {
                    for (int z = -offset_size; z <= offset_size; ++z) {
                        coord_t coordinate{center(0) + x, center(1) + y, center(2) + z};

                        auto search = sparse_map.find(coordinate);
                        if (search != sparse_map.end()) {
                            const int relative_x = x + offset_size;
                            const int relative_y = y + offset_size;
                            const int relative_z = z + offset_size;

                            output(timestep, relative_x, relative_y, relative_z) = search->second;
                        }
                    }
                }
            }
        }
    }


private:
    int max_length_;
    int window_size_;
};

REGISTER_KERNEL_BUILDER(Name("LatticeSnake").Device(DEVICE_CPU), LatticeSnakeOp);



# _merge_test_repro_0: original merge function which had produced different mean_iou for merged and non_merged in exp10-2
                      # def merge_adapters_to_base(model_id: str, adapter_dir: str, dtype=torch.bfloat16):
                      #     base_with_adapters, _ = load_model(model_id=model_id, dtype=dtype,
                      #                                        use_adapters=True, from_dir=adapter_dir)
                      #     if not isinstance(base_with_adapters, PeftModel):
                      #         raise ValueError(f"No adapters to merge at {adapter_dir}")
                      #     merged = base_with_adapters.merge_and_unload()
                      #     out = os.path.join(adapter_dir, "merged")
                      #     os.makedirs(out, exist_ok=True)
                      #     merged.save_pretrained(out, safe_serialization=True)
                      #     print(f"✅ Merged model saved to {out}")
                      #     return out
                      # results:

                      # 1 x A100 PCIe:
                      # 📊 Stage exp10_visio_warmup_merge_test_repro_0 full evaluation:
                      # {'mean_iou': 0.5687118299651845, 'precision@0.5': 0.6341463414634146, 'recall@0.5': 0.6205962059620597, 'f1@0.5': 0.6246612466124662}
                      # 📊 Stage exp10_visio_warmup_merge_test_repro_0 merged evaluation:
                      # {'mean_iou': 0.571663314244725, 'precision@0.5': 0.6260162601626016, 'recall@0.5': 0.6124661246612467, 'f1@0.5': 0.6165311653116532}

                      # 2 x A100 PCIe:
                      # 📊 Stage exp10_joint_merge_test_repro_0 full evaluation:
                      # {'mean_iou': 0.7937712586992155, 'precision@0.5': 0.8699186991869918, 'recall@0.5': 0.856368563685637, 'f1@0.5': 0.8604336043360433}
                      # 📊 Stage exp10_joint_merge_test_repro_0 merged evaluation:
                      # {'mean_iou': 0.29994166550430945, 'precision@0.5': 0.34146341463414637, 'recall@0.5': 0.33197831978319786, 'f1@0.5': 0.3346883468834689}

# _merge_test_repro_1: original with is_main added to the saving part:
                      # def merge_adapters_to_base(model_id: str, adapter_dir: str, dtype=torch.bfloat16):
                      #     accelerator = Accelerator()

                      #     base_with_adapters, _ = load_model(
                      #         model_id=model_id, dtype=dtype,
                      #         use_adapters=True, from_dir=adapter_dir
                      #     )
                      #     if not isinstance(base_with_adapters, PeftModel):
                      #         raise ValueError(f"No adapters to merge at {adapter_dir}")

                      #     merged = base_with_adapters.merge_and_unload()

                      #     if accelerator.is_main_process:
                      #         out = os.path.join(adapter_dir, "merged")
                      #         os.makedirs(out, exist_ok=True)
                      #         merged.save_pretrained(out, safe_serialization=True)
                      #         print(f"✅ Merged model saved to {out}")

                      #     accelerator.wait_for_everyone()
                      #     return os.path.join(adapter_dir, "merged")  

                      # results:
                      # couldn't fix the multy gpu issue     

 # _merge_test_repro_2: similaro to 0, with double batch size to simulate 2 gpu training. for language + vision stage, the grad_accu_step was doubled instead
                      # results:

                      # RTX with accelerator:
                      #📊 Stage exp10_joint_merge_test_repro_2 full evaluation:
                      # {'mean_iou': 0.6586830493062734, 'precision@0.5': 0.75, 'recall@0.5': 0.6916666666666667, 'f1@0.5': 0.7083333333333333}
                      # 📊 Stage exp10_joint_merge_test_repro_2 merged evaluation:
                      # {'mean_iou': 0.18592663772869855, 'precision@0.5': 0.2, 'recall@0.5': 0.16666666666666666, 'f1@0.5': 0.175}

                      # A100 with accelerator:
                      # 📊 Stage exp10_joint_merge_test_repro_2 full evaluation:
                      # {'mean_iou': 0.6646838016808033, 'precision@0.5': 0.75, 'recall@0.5': 0.6916666666666667, 'f1@0.5': 0.7083333333333333}
                      # 📊 Stage exp10_joint_merge_test_repro_2 merged evaluation:
                      # {'mean_iou': 0.17905608781147747, 'precision@0.5': 0.2, 'recall@0.5': 0.16666666666666669, 'f1@0.5': 0.175}

                      # RTX without accelerator:
                      # 📊 Stage exp10_joint_merge_test_repro_2 full evaluation:
                      # {'mean_iou': 0.6660026561468839, 'precision@0.5': 0.75, 'recall@0.5': 0.6916666666666667, 'f1@0.5': 0.7083333333333333}
                      #📊 Stage exp10_joint_merge_test_repro_2 merged evaluation:
                      # {'mean_iou': 0.1591966911451891, 'precision@0.5': 0.2, 'recall@0.5': 0.16666666666666669, 'f1@0.5': 0.175}

# _merge_test_repro_3: back to original with original batch size

                      # results:
                      #📊 Stage exp10_joint_merge_test_repro_3 full evaluation:
                      # {'mean_iou': 0.6255909010767937, 'precision@0.5': 0.7, 'recall@0.5': 0.6416666666666667, 'f1@0.5': 0
                      # 📊 Stage exp10_joint_merge_test_repro_3 merged evaluation:
                      # {'mean_iou': 0.15522485591936858, 'precision@0.5': 0.2, 'recall@0.5': 0.16666666666666666, 'f1@0.5':

# _merge_test_repro_4: back to fully changed merge function
                      # def merge_adapters_to_base(model_id: str, adapter_dir: str, dtype=torch.bfloat16):
                          # base_with_adapters, _ = load_model(model_id=model_id, dtype=dtype,
                          #                                    use_adapters=True, from_dir=adapter_dir)
                          # if hasattr(base_with_adapters, "set_adapter"):
                          #     base_with_adapters.set_adapter("default")

                          # base_with_adapters.config.use_cache = False
                          # base_with_adapters.to(torch.float32)
                          # merged = base_with_adapters.merge_and_unload()
                          # merged.to(torch.bfloat16)

                          # out = os.path.join(adapter_dir, "merged")
                          # os.makedirs(out, exist_ok=True)
                          # merged.save_pretrained(out, safe_serialization=True)
                          # print(f"✅ Merged model saved to {out}")
                          # return out
                          # results:
                          # 📊 Stage exp10_joint_merge_test_repro_5 full evaluation:
                          # {'mean_iou': 0.6885668588726501, 'precision@0.5': 0.8292682926829268, 'recall@0.5': 0.8116531165311653, 'f1@0.5': 0.8170731707317073}
                          # 📊 Stage exp10_joint_merge_test_repro_5 merged evaluation:
                          # {'mean_iou': 0.25580394109408183, 'precision@0.5': 0.2926829268292683, 'recall@0.5': 0.28319783197831977, 'f1@0.5': 0.28590785907859084}

# _merge_test_repro_5:  removed the language from stage 3    
                          # results:
                          # 📊 Stage exp10_joint_merge_test_repro_5 full evaluation:
                          # {'mean_iou': 0.6839283095990739, 'precision@0.5': 0.8130081300813008, 'recall@0.5': 0.7953929539295393, 'f1@0.5': 0.8008130081300813}  
                          # 📊 Stage exp10_joint_merge_test_repro_5 merged evaluation:
                          # {'mean_iou': 0.2707790817073508, 'precision@0.5': 0.3089430894308943, 'recall@0.5': 0.29945799457994576, 'f1@0.5': 0.30216802168021684}

# _merge_test_repro_6:  used merged stage 2 as base for stage 3
                          # results:
                          # Stage exp10_joint_merge_test_repro_6 full evaluation:
                          # {'mean_iou': 0.600913510781834, 'precision@0.5': 0.6260162601626016, 'recall@0.5': 0.6124661246612467, 'f1@0.5': 0.6165311653116532}
                          # 📊 Stage exp10_joint_merge_test_repro_6 merged evaluation:
                          # {'mean_iou': 0.596193588417349, 'precision@0.5': 0.6260162601626016, 'recall@0.5': 0.6124661246612467, 'f1@0.5': 0.6165311653116532}  

# _merge_test_repro_6: 1020-1820: doubled grad_accum_steps in all stages to simulate 2 gpu training, set load_best_model_at_end to false 
                          #  Stage exp10_joint_merge_test_repro_6 full evaluation:
                          # {'mean_iou': 0.5703192816108344, 'precision@0.5': 0.6097560975609756, 'recall@0.5': 0.5962059620596206, 'f1@0.5': 0.6002710027100272}
                          # 📊 Stage exp10_joint_merge_test_repro_6 merged evaluation:
                          # {'mean_iou': 0.5586791897847552, 'precision@0.5': 0.5934959349593496, 'recall@0.5': 0.5799457994579946, 'f1@0.5': 0.5840108401084012} 

# _merge_test_repro_7:  kept the doubled grad_accum_steps, set load_best_model_at_end to true  on 1 RTX 6000 pro
                          # results:
                          # Stage exp10_joint_merge_test_repro_7 full evaluation:
                          # {'mean_iou': 0.5702097779206129, 'precision@0.5': 0.6016260162601627, 'recall@0.5': 0.5880758807588077, 'f1@0.5': 0.5921409214092141}
                          # 📊 Stage exp10_joint_merge_test_repro_7 merged evaluation:
                          # {'mean_iou': 0.5539204210820948, 'precision@0.5': 0.5934959349593496, 'recall@0.5': 0.5799457994579946, 'f1@0.5': 0.5840108401084012}

# _merge_test_repro_8:  kept load_best_model_at_end to true, reverted grad_accum_steps to original values similar to first 6, on 2 RTX 6000 pro


### fixed the merging issue, kept load_best_model_at_end to true
    #tested 1B (grad_accum_steps to 4,8,16) vs 2B (grad_accum_steps to 8,16,32)
    # 1 GPU vs 2 GPU:

# _merge_test_repro_15: 1B, 1 GPU
                          # RTX PRO 6000 WK
                          # run 1217-1845
                          # time: 980 + 2722 + 3493 = 120 min
                          # mean_iou:
                                     # stage 1: 0.63
                                     # stage 2: 0.839
                                     # stage 3: 0.845

# _merge_test_repro_9: 1B, 2 GPU
                          # RTX PRO 6000
                          # run 1023_0003
                          # time: 24 + 31 + 42 = 97 min
                          # eval mean_iou: 0.76 --> 0.85 --> 0.87

# _merge_test_repro_10:  2B, 2 GPU
                          # RTX PRO 6000
                          # run 
                          # time: 
                          # eval mean_iou: 

# _merge_test_repro_11: 2B, 1 GPU
                          # RTX PRO 6000
                          # run 1023_1553
                          # time: 21 + 55 + 74 = 150 min
                          # eval mean_iou: 0.76 --> 0.879 --> 0.886

# _merge_test_repro_12: 1B, 2 GPU
                          # RTX PRO 6000
                          # run 1023_2035
                          # time: 15 + 31 + 43 = 89 min
                          # eval mean_iou: 0.75 --> 0.852 --> 0.893

# _merge_test_repro_14: 1B, 2 GPU, double lrs
                          # RTX PRO 6000
                          # run
                          # time:
                          # mean_iou:

# exp12:                1B, 1GPU, just stage 2
                          # time:
                          # eval mean_iou:

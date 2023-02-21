import narla


runner_settings = narla.runner.parse_args()


runner = narla.runner.Runner(
    all_settings=runner_settings.product(),
    available_gpus=runner_settings.gpus,
    jobs_per_gpu=runner_settings.jobs_per_gpu
)

runner.execute()

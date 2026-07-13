import argparse

from robopin_pipeline.common.config import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the RoboPIN PinCoT data construction pipeline.")
    parser.add_argument("--config", required=True, help="Path to pipeline JSON config.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    canonical = subparsers.add_parser("canonicalize", help="Normalize raw JSON into the shared schema.")
    canonical.add_argument("--input", required=True)
    canonical.add_argument("--output", required=True)
    canonical.add_argument("--dataset-name", required=True)

    parse = subparsers.add_parser("semantic-parse", help="Extract grounding-oriented semantic targets.")
    parse.add_argument("--input", required=True)
    parse.add_argument("--output", required=True)
    parse.add_argument("--prompt", default="prompts/parse/generic_semantic_parse_prompt.txt")

    grounding = subparsers.add_parser("grounding", help="Run Florence-2 + SAM 2 grounding.")
    grounding.add_argument("--input", required=True)
    grounding.add_argument("--output", required=True)

    xml = subparsers.add_parser("xml", help="Convert anchors into PinCoT XML-style relative points.")
    xml.add_argument("--input", required=True)
    xml.add_argument("--output", required=True)

    thinking = subparsers.add_parser("thinking", help="Generate PinCoT reasoning traces.")
    thinking.add_argument("--input", required=True)
    thinking.add_argument("--output", required=True)
    thinking.add_argument("--prompt", default="prompts/thinking/pincot_generation_prompt.txt")

    filter_cmd = subparsers.add_parser("filter", help="Filter generated traces by format and leakage checks.")
    filter_cmd.add_argument("--input", required=True)
    filter_cmd.add_argument("--output", required=True)
    filter_cmd.add_argument("--keep-rejected", action="store_true")

    full = subparsers.add_parser("full", help="Run semantic-parse -> grounding -> xml -> thinking -> filter.")
    full.add_argument("--input", required=True)
    full.add_argument("--semantic-output", required=True)
    full.add_argument("--grounding-output", required=True)
    full.add_argument("--xml-output", required=True)
    full.add_argument("--thinking-output", required=True)
    full.add_argument("--filtered-output", required=True)
    full.add_argument("--parse-prompt", default="prompts/parse/generic_semantic_parse_prompt.txt")
    full.add_argument("--thinking-prompt", default="prompts/thinking/pincot_generation_prompt.txt")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(args.config)

    if args.command == "canonicalize":
        from robopin_pipeline.stages.canonicalize import run_canonicalize

        run_canonicalize(args.input, args.output, args.dataset_name)
    elif args.command == "semantic-parse":
        from robopin_pipeline.stages.semantic_parse import run_semantic_parse

        run_semantic_parse(args.input, args.output, config, args.prompt)
    elif args.command == "grounding":
        from robopin_pipeline.stages.grounding import run_grounding

        run_grounding(args.input, args.output, config)
    elif args.command == "xml":
        from robopin_pipeline.stages.xml_stage import run_xml

        run_xml(args.input, args.output, config)
    elif args.command == "thinking":
        from robopin_pipeline.stages.thinking import run_thinking

        run_thinking(args.input, args.output, config, args.prompt)
    elif args.command == "filter":
        from robopin_pipeline.stages.filtering import run_filter

        run_filter(args.input, args.output, require_accepted=not args.keep_rejected)
    elif args.command == "full":
        from robopin_pipeline.stages.filtering import run_filter
        from robopin_pipeline.stages.grounding import run_grounding
        from robopin_pipeline.stages.semantic_parse import run_semantic_parse
        from robopin_pipeline.stages.thinking import run_thinking
        from robopin_pipeline.stages.xml_stage import run_xml

        run_semantic_parse(args.input, args.semantic_output, config, args.parse_prompt)
        run_grounding(args.semantic_output, args.grounding_output, config)
        run_xml(args.grounding_output, args.xml_output, config)
        run_thinking(args.xml_output, args.thinking_output, config, args.thinking_prompt)
        run_filter(args.thinking_output, args.filtered_output)
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()

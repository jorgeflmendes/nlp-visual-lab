export interface TraceTensor {
  name: string;
  shape: number[];
  values: number[] | Float32Array;
  dtype?: string;
  axes?: string[];
  description?: string;
}

export interface TraceToken {
  text: string;
  id?: number;
  position?: number;
}

export interface Candidate {
  token: string;
  id?: number;
  probability: number;
  logit?: number;
  selected?: boolean;
}

export interface ExecutionStep {
  name: string;
  stage: string;
  operation: string;
  description: string;
  formula?: string;
  tensors: TraceTensor[];
  tokens?: TraceToken[];
  selectedToken?: number;
  probabilities?: Candidate[];
  attention?: { source: string[]; target: string[]; weights: number[][]; row: number };
}

export interface ExecutionTrace {
  output: string;
  outputLabel: string;
  backend: string;
  elapsed: number;
  steps: ExecutionStep[];
  note?: string;
}

# Timeout Issue - Visual Guide

## Current Problem: Timeout Timeline

```
CURRENT CONFIGURATION (120-second timeout):

qwen3:32b Request (needs ~200 seconds):
  0s     Request starts
  30s    Model loading complete
  60s    Processing half-way
  90s    Processing 75% done
  120s   [TIMEOUT ERROR]  Request fails!
  (110 seconds of potential work lost)

qwen3:14b Request (needs ~150 seconds):
  0s     Request starts
  30s    Model loading complete
  60s    Processing 40% done
  90s    Processing 60% done
  120s   [TIMEOUT ERROR]  Request fails!
  (30 seconds of potential work lost)

qwen3:8b Request (needs ~60 seconds):
  0s     Request starts
  30s    Model loading complete
  60s    [SUCCESS]  Completes just in time!
```

## Fixed Solution: Extended Timeout Timeline

```
RECOMMENDED CONFIGURATION (600-second timeout):

qwen3:32b Request (needs ~200 seconds):
  0s     Request starts
  30s    Model loading complete
  60s    Processing 30% done
  120s   Processing 60% done
  180s   Processing 90% done
  200s   [SUCCESS]  Completes!
  (Plenty of buffer time remaining: 400 seconds)

qwen3:14b Request (needs ~150 seconds):
  0s     Request starts
  30s    Model loading complete
  60s    Processing 40% done
  120s   Processing 80% done
  150s   [SUCCESS]  Completes!
  (Plenty of buffer time remaining: 450 seconds)

qwen3:8b Request (needs ~60 seconds):
  0s     Request starts
  30s    Model loading complete
  60s    [SUCCESS]  Completes quickly!
  (Plenty of buffer time remaining: 540 seconds)
```

## Architecture: How Timeout Flows Through Code

```

                      User Config                             
                    (config.yaml)                             
              llm_timeout: 120.0 seconds                      

                         
                         

               Config Loading (config.py)                     
    load_config()  get_float("llm_timeout", 120.0)          
                                                              
    Issue: Fallback default (120.0) conflicts with          
    dataclass definition (900.0)                             

                         
                         

              Config Object Created                           
         config.llm_timeout = 120.0 seconds                  

                         
                         

           Provider Factory (factory.py)                      
    kwargs["timeout"] = getattr(config,                      
                        "llm_timeout", 120.0)                
                      = 120.0 seconds                        

                         
                         

          Provider Initialization                             
   OllamaProvider(timeout=120.0)  [overrides 900.0 default]  
   httpx.Client(timeout=httpx.Timeout(120.0))               

                         
                         

           Request Execution (ollama.py)                      
    response = client.post(..., timeout=120.0)              
                                                              
    For qwen3:32b (needs 200s):                             
     120 seconds elapsed  httpx.TimeoutException        

```

## Timeout Configuration Hierarchy

```
PRECEDENCE ORDER (highest to lowest):


 1. Environment Variable (Highest)       
    export LLM_TIMEOUT=600.0             
    Overrides everything                 

               
               

 2. Config File (config.yaml)            
    llm_timeout: 600.0                   
    Overrides code defaults              

               
               

 3a. Config Load Fallback (config.py)    
     get_float("llm_timeout", 120.0)     
     Used if not in config file          

               
               

 3b. Dataclass Default                   
     Config.llm_timeout = 900.0          
     NOT USED (overridden by 3a)         

               
               

 4. Provider Defaults (Lowest)           
    OllamaProvider(timeout=900.0)        
    Never reached (config value used)    

```

## Model Performance Expectations

```
HARDWARE PERFORMANCE ESTIMATES:
(Apple Silicon M-series CPU assumptions)

Model          Size    Load Time   Eval Time   Total

qwen3:8b       8B      10-15s      30-50s      40-65s
qwen3:14b      14B     20-30s      60-150s     80-180s  TIMEOUT at 120s!
qwen3:32b      32B     30-40s      120-260s    150-300s  TIMEOUT at 120s!
llama3:70b     70B     45-60s      200-400s+   245-460s+  TIMEOUT at 120s!

Note: Times vary by prompt length, model quantization, and hardware specs
```

## Issue vs Solution Comparison

```

                   CURRENT (BROKEN)  FIXED            

 Config Default    120.0 seconds     900.0 seconds    
 Config File       120.0 seconds     600.0 seconds    
 Ollama Provider   900.0 (overridden) 900.0 (used)    
 LM Studio         120.0 seconds     600.0 seconds    
 OpenRouter        120.0 seconds     180.0 seconds    

 qwen3:8b Result    Works (60s)      Works (60s)    
 qwen3:14b Result   TIMEOUT         Works (150s)   
 qwen3:32b Result   TIMEOUT         Works (200s)   

```

## Retry Logic Impact

```
RETRY MECHANISM BEHAVIOR WITH 120-SECOND TIMEOUT:

Request Timeline:

  Attempt 1:
   Start: 0s
   Processing: 0-120s
   Timeout at 120s 
  
  Wait 2s (retry delay)
  
  Attempt 2:
   Start: 122s
   Processing: 122-242s
   Timeout at 242s 
  
  Wait 4s (retry delay)
  
  Attempt 3:
   Start: 246s
   Processing: 246-366s
   Timeout at 366s 
  
  FAILURE: All 3 attempts exhausted

  Key point: Retry mechanism CANNOT help when timeout is the root cause!
    Each attempt still hits the same 120-second limit.
    Retry is for transient failures (network glitches), not timeouts.
```

## Configuration Priority in Code

```python
# config.py lines 179-181
def get_float(key: str, default: float) -> float:
    val = config_data.get(key) or os.getenv(key.upper())
    #      Check YAML first     Check ENV if not in YAML
    return float(val) if val is not None else default
    #                                            Fallback if neither found

# So the check order is:
# 1. config.yaml: llm_timeout: 120.0     HIGHEST (currently used)
# 2. LLM_TIMEOUT=600.0 (env var)         Would override YAML
# 3. get_float default: 120.0             Used only if not in YAML or ENV
# 4. Config dataclass: 900.0              NEVER USED (overridden above)
```

## Summary: Why This Breaks

```
PROBLEM CHAIN:

1. Config loads with 120s default (config.py:213)
   
    config.yaml explicitly sets 120.0
     (has been this way since initial setup)
   
2. Timeout value (120s) passed to provider factory
   
    Factory overrides Ollama provider's 900s default
     (provider default is never used)
   
3. Ollama client initialized with 120s timeout
   
    httpx.Timeout(120.0) created
   
4. Request to qwen3:32b (needs ~200s)
   
    Request times out after exactly 120 seconds
      User sees error, card generation fails
```

## Solution Chain

```
SOLUTION CHAIN:

1. Update config.py fallback from 120.0 to 900.0
    Aligns with dataclass definition

2. Update config.yaml from 120.0 to 600.0  
    Explicitly sets appropriate value

3. Update example configs to 600.0
    Helps future users avoid the same issue

4. Update provider defaults to 600.0 (local) or 180.0 (cloud)
    Reasonable timeouts for their deployment model

5. Result: qwen3:32b and 14b requests succeed! 
```


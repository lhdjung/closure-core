Yes, please implement this plan, with the following decisions made regarding your "Key Differences to Address":

1. Return Type Mismatch: SPRITE streaming should do all the same as CLOSURE streaming.
2. Statistics Computation: I disagree. Instead, SPRITE should stream the exact same statistics as CLOSURE; including horns and frequency statistics. In fact, I just renamed ClosureResult to ResultListFromMeanSdN to reflect the more general nature of this data structure: it should encompass the results of both CLOSURE and SPRITE.
3. File Structure: SPRITE files should be all the same as CLOSURE files. Therefore, forget about "statistics.parquet" -- CLOSURe doesn't have it, so SPRITE shouldn't have it, either.

## To the rest of your plan

Yes, please enhance sprite_parallel() with ParquetConfig. The same basic idea applies: CLOSURE has it, so SPRITE should have it, too. The API should be just the same. Follow CLOSURE patterns.

Reuse existing patterns as much as possible. Don't add anything that is not present in the CLOSURE implementation.

## To your open questions

Very simple: Do all that CLOSURE does, but don't do anything else. In particular, don't add any deduplication features for now.

In summary, I want the output of SPRITE to be just like the output of CLOSURE. This is very important for consistency. The ultimate goal is that the crate's API is just the same for SPRITE as for CLOSURE.
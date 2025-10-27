
# breath is a data.frame with columns: patient | voc1 | ... | voc_xyz
# blanks is a data.frame with columns: patient | voc1 | ... | voc_xyz
# params is a list to ensure no leakage when moving from train -> test
# set.
incorporate.blanks <- function(breath, blanks, params = NULL){
  # This will only output the patient concatenated with one observation for each VOC.
  breath <- breath %>% select(patient, starts_with('voc'))
  blanks <- blanks %>% select(patient, starts_with('voc'))
  stopifnot(all(breath$patient == blanks$patient))
  
  # Check ordering correct ... 
  if (!is.null(params)) {
    stopifnot(all.equal(names(breath)[-1], names(blanks)[-1], names(params$medians)))
  }
  
  # bl; br matrices of intensities
  bl <- as.matrix(blanks[, -1, drop = F])
  br <- as.matrix(breath[, -1, drop = F])
  if(all(bl == 0)){
    # If all blanks are set to zero (my dummy chosen value) then do the below
    # sub-routine
    temp <- incorporate.no.blanks(br, params = params)
    return(
      list(
        data = as_tibble(data.frame(patient = breath$patient, temp$out)), 
        mfc.only = as_tibble(as.data.frame(temp$mfc.only)), 
        corrected = as_tibble(as.data.frame(temp$corrected)),
        params = temp$params
      )
    )
  }
  # raw ratio
  rat <- br/(bl + 1.)
  
  # MFC normalisation
  meds <- if(is.null(params)) apply(rat, 2, median) else params$medians
  nk <- apply(rat, 1, \(x) median(x/meds))
  norm <- sweep(rat, 1, nk, '/')
  
  # log variance stabilising
  eps <- if(is.null(params)) median(norm[norm > 0]) else params$eps
  vst <- log(norm + eps)
  
  out <- data.frame(patient = breath$patient, vst) %>% as_tibble
  params <- list(medians = meds, eps = eps)
  
  # Return list of data and data-specific items
  return(list(
    data = out, corrected = rat, mfc.only = norm, params = params
  ))
}

# sub-routine for case where we only have raw breath intensities
incorporate.no.blanks <- function(breath, params = NULL){
  # Targetting median profile
  meds <- if(is.null(params)) apply(breath, 2, \(x) median(x[x > 0])) else params$medians
  # Per-sample scale via median fold change against target median profile
  nk <- apply(breath, 1, \(x) median(x/meds))
  norm <- sweep(breath, 1, nk, '/') 
  eps <- if(is.null(params)) median(norm[norm>0]) else params$eps
  vst <- log(abs(norm) + eps)
  params <- list(offset = 0, medians = meds, eps = eps)
  list(
    out = vst, mfc.only = norm, params = params, corrected = log1p(breath)
  )
}